def main():
    import os
    from guardgraph.graph import InteractionsGraph
    # If you need to re-initiate delete/move dir /data/globi
    if not os.path.exists('/data/globi'):
        os.mkdir('/data/globi')
        ig = InteractionsGraph()
        ig.retrieve_data_files()
        ig.prep_admin_import_files()
