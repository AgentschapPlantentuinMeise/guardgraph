import os
import gzip
from neo4j import GraphDatabase

class InteractionsGraph(object):
    def __init__(self, password=None, passfile='/data/.neo4j_credentials'):
        self._passfile = passfile
        if password: self._password = password
        elif os.path.exists(self._passfile):
            self._password = open(self._passfile).read()
        else:
            self._password = self.set_random_password()
        self.driver = GraphDatabase.driver("neo4j://localhost:7687",
                              auth=("neo4j", self._password))

    def __del__(self):
        self.driver.close()

    # neo4j admin methods
    def set_password(self, password):
        driver = GraphDatabase.driver("neo4j://localhost:7687",
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

    # graph loading methods
    def load_interaction(self, interactions_file='/data/globi/interactions.tsv.gz'):
        intergz = gzip.open('/data/globi/interactions.tsv.gz')
        header = intergz.readline().decode().strip().split('\t')
        for line in intergz:
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
            with self.driver.session(database="neo4j") as session:
                session.execute_write(self.add_interaction, "species1", "species2")
        session.execute_read(print_friends, "species1")

    @static_method
    def add_interaction(
            tx, source_name, source_id, source_rank,
            target_name, target_id, target_rank, ix_name):
        escaped_ix_name = ix_name.capitalize().replace("\\u0060", "`").replace("`", "``")
        tx.run("MERGE (source:Taxon {name: $source_name, id: $source_id, rank: $source_rank}) "
               "MERGE (target:Taxon {name: $target_name, id: $target_id, rank: $target_rank}) "
               f"MERGE (source)-[:`{escaped_ix_name}`]->(target)",
               source_name=source_name, source_id=source_id, source_rank=source_rank,
               target_name=target_name, target_id=target_id, target_rank=target_rank,
               ix_name=ix_name
        )

    @static_method
    def print_interactions(tx, name):
        query = ("MATCH (a:Species)-[:KNOWS]->(species) WHERE a.name = $name "
             "RETURN species.name ORDER BY species.name")
        for record in tx.run(query, name=name):
            print(record["species.name"])
            
