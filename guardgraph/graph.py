import os
import gzip
from neo4j import GraphDatabase
from itertools import count

class InteractionsGraph(object):
    def __init__(self, password=None, passfile='/data/.neo4j_credentials'):
        self._passfile = passfile
        if password: self._password = password
        elif os.path.exists(self._passfile):
            self._password = open(self._passfile).read().strip()
        else:
            self._password = self.set_random_password()
        self.driver = GraphDatabase.driver("neo4j://neo4j:7687",
                              auth=("neo4j", self._password))

    def __del__(self):
        self.driver.close()

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

    def reset_graph(self, force=False):
        "Deletes all current nodes and relationships"
        if not force:
            if not input('Are you sure you would to delete the full graph? ') == 'yes':
                return
        with self.driver.session(database="neo4j") as session:
            session.execute_write(lambda tx: tx.run("MATCH (n) DETACH DELETE n;"))

    # prep_interaction_data
    def prep_interaction_data(self, interactions_file='/data/globi/interactions.tsv.gz'):
        c = count()
        with gzip.open(interactions_file) as intergz:
            for line in intergz:
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
    #@staticmethod
    def load_taxons(self):#tx, *args):
        #zcat data/globi/interactions.tsv.gz | sed 's/"/""/g' > import interactions.tsv
        q = '''
        LOAD CSV WITH HEADERS FROM 'file:///interactions.tsv' AS line FIELDTERMINATOR '\t'
        //WITH line LIMIT 10
        WITH line WHERE line.sourceTaxonRank IS NOT NULL AND line.targetTaxonRank IS NOT NULL
        CALL {
          WITH line
          MERGE (:Taxon {name: line.sourceTaxonName, id: line.sourceTaxonId, rank: line.sourceTaxonRank})
          MERGE (:Taxon {name: line.targetTaxonName, id: line.targetTaxonId, rank: line.targetTaxonRank})
        } IN TRANSACTIONS OF 5000 ROWS
        '''
        #result = tx.run(q)
        #return result.data()
        with self.driver.session(database="neo4j") as session:
            session.run(q)
        
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

