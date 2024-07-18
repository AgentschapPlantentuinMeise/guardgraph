def main():
    import os
    from guardgraph.graph import InteractionsGraph
    os.mkdir('/data/globi')
    ig = InteractionsGraph()
    ig.retrieve_data_files()
    ig.prep_admin_import_files()
