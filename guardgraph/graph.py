import os
import re
import time
import gzip
import logging
import warnings
from itertools import count
import pandas as pd
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
from guardgraph.utils import download_file

class InteractionsGraph(object):
    def __init__(self, password=None, passfile='/data/.neo4j_credentials', initialize_database=False):
        self._passfile = passfile
        if password: self._password = password
        elif os.path.exists(self._passfile):
            self._password = open(self._passfile).read().strip()
        else:
            self._password = self.set_random_password()
        self.driver = GraphDatabase.driver("neo4j://neo4j:7687",
                              auth=("neo4j", self._password))
        if initialize_database:
            self.initialize_database()

    def __del__(self):
        self.driver.close()

    def run_query(self, q, database="neo4j", timeit=False, **kwargs):
        "Run query directly through session (implicit commit)"
        with self.driver.session(database=database) as session:
            if timeit:
                start = time.perf_counter()
            result = session.run(q, **kwargs)
            if timeit:
                logging.info('Query took %s s', time.perf_counter()-start)
            return result.data()
            
    # neo4j admin methods
    def set_password(self, password):
        driver = GraphDatabase.driver("neo4j://neo4j:7687",
                              auth=("neo4j", "neo4j"))
        with driver.session(database="system") as session:
            result = session.execute_write(
                lambda tx: tx.run(f"ALTER CURRENT USER SET PASSWORD FROM 'neo4j' TO '{password}';"))
        driver.close()

    def set_random_password(self):
        import random
        import string
        password = ''.join(random.choice(
            string.ascii_letters + string.digits + string.punctuation
        ) for i in range(16))
        with open(self._passfile, 'wt') as out:
            out.write(password)
        self.set_password(password)
        return password

    def reset_graph(self, force=False, relationships_only=False):
        "Deletes all current nodes and relationships"
        if not force:
            if not input('Are you sure you would to delete the full graph? ') == 'yes':
                return
        with self.driver.session(database="neo4j") as session:
            # In neo4j browser:
            # :auto MATCH ()-[r]->() CALL { WITH r DELETE r } IN TRANSACTIONS OF 10000 ROWS
            session.run("MATCH ()-[r]->() CALL { WITH r DELETE r } IN TRANSACTIONS OF 10000 ROWS;")
            if not relationships_only:
                # Relationships should already be removed, but running this
                # without deleting rel's separately gave mem issues
                session.run("MATCH (n) CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS;")

    def initialize_database(self):
        """Initialize the full database.
        Should do all steps required to build up database from scratch
        """
        # Requirement: globi interactions file should have been downloaded
        # TODO automate download
        self.prep_interaction_data_file()
        try:
            self.query('create_name_index', write=True)
        except Neo4jError as e:
            warnings.warn('Indices already created')
        self.load_taxons()
        self.load_taxon_labels()
        self.load_taxon_relationships()

    # retrieve files
    def retrieve_data_files(self):
        """Retrieve all required files for building up the graph"""
        download_file(
            'https://zenodo.org/record/7348355/files/interactions.tsv.gz',
            '/data/globi'
        )
        self.prep_interaction_data_file()
        download_file(
            'https://zenodo.org/record/7348355/files/taxonCache.tsv.gz',
            '/data/neo4j_import'
        )
    
    # prep_interaction_data
    def prep_interaction_data_file(self, interactions_file='/data/globi/interactions.tsv.gz'):
        """neo4j LOAD CSV has an issue with quotes
        hence they are removed and the file is written
        to the location were neo4j can find it

        bash command: zcat data/globi/interactions.tsv.gz | sed 's/"//g' > import/interactions.tsv

        With current preprocessing takes around 10 min
        Tested it with both byte and decoded but only made a few seconds difference
        """
        c = count()
        #pattern = re.compile(r'"') # +- 4x longer than string replace
        if not os.path.exists('/data/neo4j_import'):
            os.mkdir('/data/neo4j_import')
            print("Created neo4j import dir for volume mapping")
        with gzip.open(interactions_file, 'rt') as intergz:
            with gzip.open('/data/neo4j_import/globi.tsv.gz', 'wt') as gz_out:
                for line in intergz:
                    gz_out.write(line.replace('"',''))
                    #gz_out.write(pattern.sub('', line))
                    next(c)
        print(c)

    @staticmethod
    def prep_interaction_data_line(line):
        line = line.decode().strip().split('\t')
        sourceTaxonId = line[0]
        ## Properties
        sourceTaxonName = line[2]
        sourceTaxonRank = line[3]
        sourceOccurenceId = line[24]
        # Relationship
        interactionTypeName = line[38]
        decimalLatitude = line[78]
        decimalLongitude = line[79]
        localityId = line[80]
        localityName = line[81]
        eventDate = line[82]
        # Target
        targetTaxonId = line[40]
        ## Properties
        targetTaxonName = line[42]
        targetTaxonRank = line[43]
        targetOccurenceId = line[64]
        processed_line = {
            'source_name':sourceTaxonName,
            'source_id': sourceTaxonId,
            'source_rank': sourceTaxonRank.capitalize() or 'NA',
            'target_name': targetTaxonName,
            'target_id': targetTaxonId,
            'target_rank': targetTaxonRank.capitalize() or 'NA',
            'ix_name': interactionTypeName,
            'escaped_ix_name': interactionTypeName.upper(
            ).replace("\\u0060", "`").replace("`", "``")
        }
        return processed_line
            
    # graph loading methods
    def load_taxons(self):
        # Transactions requires implicit committing
        # Last timeit: 1h 1min 57s
        q = '''
        LOAD CSV WITH HEADERS FROM 'file:///globi.tsv.gz' AS line FIELDTERMINATOR '\t'
        //WITH line LIMIT 10
        WITH line WHERE line.sourceTaxonRank IS NOT NULL AND line.targetTaxonRank IS NOT NULL
        CALL {
          WITH line
          MERGE (source:Taxon {name: line.sourceTaxonName})
          ON CREATE
            SET source.ids = [line.sourceTaxonId],
                source.rank = line.sourceTaxonRank
          ON MATCH
            SET source.ids = CASE WHEN line.sourceTaxonId IN source.ids
                THEN source.ids ELSE source.ids + line.sourceTaxonId END
          MERGE (target:Taxon {name: line.targetTaxonName})
          ON CREATE
            SET target.ids = [line.targetTaxonId],
                target.rank = line.targetTaxonRank
          ON MATCH
            SET target.ids = CASE WHEN line.targetTaxonId IN target.ids
                THEN target.ids ELSE target.ids + line.targetTaxonId END
        } IN TRANSACTIONS OF 5000 ROWS
        '''
        # put rank in create step
        self.run_query(q)

    def load_taxon_labels(self):
        # Last timeit: 1h 2min 9s
        q = '''
        LOAD CSV WITH HEADERS FROM 'file:///globi.tsv.gz' AS line FIELDTERMINATOR '\t'
        //WITH line LIMIT 10
        WITH line WHERE line.sourceTaxonRank IS NOT NULL AND line.targetTaxonRank IS NOT NULL
        CALL {
          WITH line
          MATCH (s:Taxon {name: line.sourceTaxonName})
          WITH s, line
          CALL apoc.create.addLabels(s, [line.sourceTaxonRank]) YIELD node as nSource
          MATCH (t:Taxon {name: line.targetTaxonName})
          WITH t, line
          CALL apoc.create.addLabels(t, [line.targetTaxonRank]) YIELD node as tSource
          RETURN tSource
        } IN TRANSACTIONS OF 1000 ROWS
        RETURN COUNT(tSource)
        '''
        self.run_query(q)

        
    def load_taxon_relationships(self):
        # Last timeit: 1h 9min 22s
        # No time difference whether setting only occurences or with ref doi
        q = '''
        LOAD CSV WITH HEADERS FROM 'file:///globi.tsv.gz' AS line FIELDTERMINATOR '\t'
        WITH line WHERE line.sourceTaxonRank IS NOT NULL AND line.targetTaxonRank IS NOT NULL
        CALL {
          WITH line
          MATCH (s:Taxon {name: line.sourceTaxonName})
          MATCH (t:Taxon {name: line.targetTaxonName})
          WITH s, t, line
          CALL apoc.merge.relationship(
            s, line.interactionTypeName, NULL,
            {occurences: 0, references: []},
            t, {}
          ) YIELD rel SET rel.occurences = rel.occurences+1,
                          rel.references = rel.references+line.referenceDoi
          RETURN rel
        } IN TRANSACTIONS OF 5000 ROWS
        RETURN COUNT(rel)
        '''
        # occurence id (not always), doi (publication), date, dataset (how it got into globi)
        with self.driver.session(database="neo4j") as session:
            session.run(q)
        # TIME started 10:34, ended 
        
    def load_interaction_data(self, interactions_file='/data/globi/interactions.tsv.gz', start=0, max_entries=10, batch_size=1, cypher_batched=True):
        if max_entries: entries_count = count()
        intergz = gzip.open(interactions_file)
        header = intergz.readline().decode().strip().split('\t')
        print(header)
        batch = []
        for line in intergz:
            current_entry = next(entries_count)
            if max_entries and max_entries+start-1 < current_entry:
                break
            elif start and current_entry < start: continue
            batch.append(self.prep_interaction_data_line(line))
            if len(batch) > batch_size:
                with self.driver.session(database="neo4j") as session:
                    if cypher_batched:
                        session.execute_write(self.add_interactions,
                                              interactions=batch)
                    else:
                        for interaction in batch:
                            session.execute_write(
                                self.add_interaction, **interaction)
                batch = []
        if batch:
            with self.driver.session(database="neo4j") as session:
                if cypher_batched:
                    session.execute_write(self.add_interactions,
                                          interactions=batch)
                else:
                    for interaction in batch:
                        session.execute_write(
                            self.add_interaction, **interaction)
        intergz.close()
        self.query('count_interactions')

    def query(self, query_method, *args, write=False, **kwargs):
        with self.driver.session(database="neo4j") as session:
            result = (session.execute_write if write else
                session.execute_read)(
                    self.__getattribute__(query_method), *args, **kwargs
            )
            print(result)

    @staticmethod
    def create_name_index(tx, *args):
        q = "CREATE INDEX node_range_index_name FOR (n:Taxon) ON (n.name)"
        result = tx.run(q)
        return result.data()
            
    @staticmethod
    def add_interaction(
            tx, source_name, source_id, source_rank,
            target_name, target_id, target_rank,
            ix_name, escaped_ix_name):
        tx.run("MERGE (source:Taxon {name: $source_name, id: $source_id, rank: $source_rank}) "
               "MERGE (target:Taxon {name: $target_name, id: $target_id, rank: $target_rank}) "
               f"MERGE (source)-[:`{escaped_ix_name}`]->(target)",
               source_name=source_name, source_id=source_id, source_rank=source_rank,
               target_name=target_name, target_id=target_id, target_rank=target_rank,
               ix_name=ix_name
        )

    @staticmethod
    def add_interactions(tx, interactions):
        query = '''
          UNWIND $rows AS row
          MERGE (source:Taxon {name: row.source_name, id: row.source_id, rank: row.source_rank})
          WITH source, row
          CALL apoc.create.addLabels(source, [row.source_rank]) YIELD node
          MERGE (target:Taxon {name: row.target_name, id: row.target_id, rank: row.target_rank})
          WITH source, target, row
          CALL apoc.create.relationship(source, row.escaped_ix_name, NULL, target) YIELD rel
          RETURN count(*) as total
        '''
        tx.run(query, rows=interactions)

    @staticmethod
    def print_interactions(tx, name):
        query = ("MATCH (a:Taxon)-[:KNOWS]->(taxon) WHERE a.name = $name "
             "RETURN taxon.name ORDER BY taxon.name")
        for record in tx.run(query, name=name):
            print(record["taxon.name"])
            
    @staticmethod
    def count_interactions(tx, *args): #TODO with no needed args still requires *args
        query = "MATCH ()-[r]->() RETURN COUNT(r);"
        result = tx.run(query)
        return result.data()

class EcoAnalysis(object):
    # %pip install multimethod tqdm
    # %pip install --no-deps graphdatascience
    def __init__(self, interactiongraph=None):
        from graphdatascience import GraphDataScience
        self.ig = InteractionGraph() if interactiongraph is None else interactiongraph
        self.gds = GraphDataScience(
            'neo4j://neo4j:7687',
            auth=('neo4j', self.ig._password)
        )

    def create_projection(self, name, node_projection=None, relationship_projection=None):
        """Create graph data model (projection)

        Example:
          >>> ea = EcoAnalysis(ig)
          >>> ea.create_projection(
          ...     'test_projection',
          ...     ['species','genus'],
          ...     {
          ...       'pollinates':{'properties':['occurences']},
          ...       'eats':{'properties':['occurences']}
          ...     }
          ... )
        """
        self.node_projection = ['Taxon'] if node_projection is None else node_projection
        self.relationship_projection = {
            'eats': {'orientation': 'NATURAL'}
        } if relationship_projection is None else relationship_projection
        result = self.gds.graph.project.estimate(
            self.node_projection,
            self.relationship_projection
        )
        print(result['requiredMemory'])
        if input('Continue? ') != 'yes':
            return
        
        G, result = self.gds.graph.project(
            name, self.node_projection,
            self.relationship_projection
        )
        print(f"The projection took {result['projectMillis']} ms")
        print(f"Graph '{G.name()}' node count: {G.node_count()}")
        print(f"Graph '{G.name()}' node labels: {G.node_labels()}")
        try:
            self.projections[name] = G
        except AttributeError:
            self.projections = {}
            self.projections[name] = G

    def create_embedding(self, projection, embeddingDimension=4):
        result = self.gds.fastRP.mutate.estimate(
            self.projections[projection],
            mutateProperty="embedding",
            randomSeed=42,
            embeddingDimension=embeddingDimension,
            relationshipWeightProperty="occurences",
            iterationWeights=[0.8, 1, 1, 1],
        )
        print(f"Required memory for running FastRP: {result['requiredMemory']}")

        if input('Continue? ') != 'yes':
            return

        result = self.gds.fastRP.mutate(
            self.projections[projection],
            mutateProperty="embedding",
            randomSeed=42,
            embeddingDimension=embeddingDimension,
            relationshipWeightProperty="occurences",
            iterationWeights=[0.8, 1, 1, 1],
        )
        print(f"Number of embedding vectors produced: {result['nodePropertiesWritten']}")
        
        # Get embeddings
        embeddings = self.gds.run_cypher(
            f"""
            CALL gds.graph.nodeProperty.stream('{projection}', 'embedding')
            YIELD nodeId, propertyValue
            RETURN gds.util.asNode(nodeId).name AS name, propertyValue AS embedding
            ORDER BY embedding DESC LIMIT 100
            """
        )
        return embeddings

    def get_embeddings(self, projection, species_list=None, plot=False):
        """Get the embeddings for projection
        optionally only for the species in species_list
        """
        if species_list:
            embeddings = self.ig.run_query( #gds.run_cypher(
            f"""
            CALL gds.graph.nodeProperty.stream('{projection}', 'embedding')
            YIELD nodeId, propertyValue
            WITH gds.util.asNode(nodeId).name AS name, propertyValue AS embedding
            WHERE name IN $species_list
            RETURN name, embedding
            ORDER BY embedding
            """, species_list=species_list
            )
            embeddings = pd.DataFrame(embeddings)
        else:
            embeddings = self.gds.run_cypher(
            f"""
            CALL gds.graph.nodeProperty.stream('{projection}', 'embedding')
            YIELD nodeId, propertyValue
            WITH gds.util.asNode(nodeId).name AS name, propertyValue AS embedding
            RETURN name, embedding
            """
            )
        return embeddings
        
    def predict_similars(self, projection):
        result = self.gds.knn.write(
            self.projections[projection],
            topK=2,
            nodeProperties=["embedding"],
            randomSeed=42,
            concurrency=1,
            sampleRate=1.0,
            deltaThreshold=0.0,
            writeRelationshipType="SIMILAR",
            writeProperty="score",
        )

        print(f"Relationships produced: {result['relationshipsWritten']}")
        print(f"Nodes compared: {result['nodesCompared']}")
        print(f"Mean similarity: {result['similarityDistribution']['mean']}")

        #Exploring similar results
        return self.gds.run_cypher(
            """
            MATCH (t1:Taxon)-[r:SIMILAR]->(t2:Taxon)
            RETURN t1.name AS taxon1, t2.name AS taxon2, r.score AS similarity
            ORDER BY similarity DESCENDING, taxon1, taxon2
            LIMIT 5
            """
        )

    def get_recommendations(self, projection):
        #Get potential menu
        # self.gds.run_cypher(
        #     """
        #         MATCH (:Taxon {name: "Acronicta impleta"})-[:eats]->(t1:Taxon)
        #         WITH collect(t1) as dishes
        #         MATCH (:Taxon {name: "Orgyia definita"})-[:eats]->(t2:Taxon)
        #         WHERE not t2 in dishes
        #         RETURN t2.name as recommendation
        #     """
        # )
        pass

    def drop_projection(self, projection):
        # Remove projection from the GDS graph catalog
        self.projections[projection].drop()

# Triad analysis
# ig.run_query("MATCH p=((n:species)-[]-(:species)-[]-(:species)-[]-(n)) RETURN p LIMIT 1")
# triad_types = ig.run_query("MATCH (n:species)-[a]-(:species)-[b]-(:species)-[c]-(n) RETURN TYPE(a),TYPE(b),TYPE(c) LIMIT 1")
# triad_types = pd.DataFrame(triad_types)
# triad_types.sum(axis=1).value_counts()
