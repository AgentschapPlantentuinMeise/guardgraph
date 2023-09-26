import os
import random
import zipfile
import pickle
import joblib
import numpy as np
import pandas as pd
import urllib.request
import requests
from json import JSONDecodeError
import tqdm
from scipy.stats import spearmanr, pearsonr
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
from guardgraph.graph import InteractionsGraph, EcoAnalysis

# Analysis settings
presence_only = False
model_type = 'environmental' # environmental|spatial
species_subset_frac = .5 # subsample species for prototyping
observations_subset_frac = False #.8 # subsample observations for prototyping
pca_fit_on_subset = True # training PCA on subset to test for bias of informing training data with PCA of all predictions that was informed by full truth set
test_randomised_embeddings = False # Randomise embeddings to test relevance


# Example code
# https://www.kaggle.com/code/histoffe/baseline-spatial-rf-pa-sum

# Kaggle data
if presence_only:
    ## Presence only
    pozf = zipfile.ZipFile('/data/Presence_only_occurrences.zip')
    f=pozf.open('Presence_only_occurrences/Presences_only_train.csv')
    presence_data = pd.read_csv(f, sep=';')
    observations_per_species = presence_data.gbifID.value_counts()

else:
    ## Presense absence observations per species
    pa_train_file = "/data/Presences_Absences_train.csv"
    if not os.path.exists(pa_train_file):
        urllib.request.urlretrieve(
            "https://lab.plantnet.org/seafile/d/936fe4298a5a4f4c8dbd/files/?p="
            "%2FPresence_Absence_surveys%2FPresences_Absences_train.csv&dl=1",
            pa_train_file
        )
    pa_test_file = "/data/PA_test_blind.csv"
    if not os.path.exists(pa_train_file):
        urllib.request.urlretrieve(
            "https://lab.plantnet.org/seafile/f/10286ade1f6b499fadae/?dl=1",
            pa_test_file
        )
    ### Load files
    #### Training data
    pa_train = pd.read_csv(pa_train_file, sep=';')
    pa_train = pa_train.groupby(['patchID', 
                                 'lon',
                                 'lat',
                                 'speciesId'
    ]).first().reset_index()
    pa_train['tmpId'] = np.core.defchararray.add(
        [str(el) + '_' for el in pa_train['patchID']],
        [str(el) for el in pa_train['dayOfYear']]
    )
    pa_species = {str(s) for s in pa_train.speciesId.unique()}
    
    multi_label_pa_train = pa_train[['patchID', 'lon', 'lat', 'speciesId']]
    multi_label_pa_train['label'] = 1
    multi_label_pa_train = pd.pivot(multi_label_pa_train,
        index=['lat', 'lon', 'patchID'],
        columns='speciesId',
        values='label').reset_index().fillna(0)
    ##### Subsample for running locally
    if species_subset_frac and species_subset_frac < 1:
        multi_label_pa_train = multi_label_pa_train.sample(
            frac=species_subset_frac, axis=1
        )
        pa_train = pa_train[
            pa_train.speciesId.isin(multi_label_pa_train.columns)
        ].copy()
        pa_species = {
            str(s) for s in pa_train.speciesId.unique()
        }
    if observations_subset_frac and observations_subset_frac < 1:
        multi_label_pa_train = multi_label_pa_train.sample(
            frac=observations_subset_frac
        )
        
    #### Test
    pa_test = pd.read_csv(pa_test_file, sep=';')
    pa_test = pa_test.groupby(['patchID', 
                                 'lon',
                                 'lat',
                                 #'speciesId'
    ]).first().reset_index()
    pa_test['tmpId'] = np.core.defchararray.add(
        [str(el) + '_' for el in pa_test['patchID']],
        [str(el) for el in pa_test['dayOfYear']]
    )
    
    #multi_label_pa_test = pa_test[['patchID', 'lon', 'lat']]#, 'speciesId']]
    #multi_label_pa_test['label'] = 1
    #multi_label_pa_test = pd.pivot(multi_label_pa_test,
    #    index=['lat', 'lon', 'patchID'],
    #    columns='speciesId',
    #    values='label').reset_index().fillna(0)

## Enivronmental and spatial data
zf = zipfile.ZipFile('/data/inra-public-archivedwl-903.zip')
files=zf.infolist()
list(filter(lambda x: '/' not in x.filename, files))
if model_type == 'environmental':
    ## Train/test data
    env_train_file = zf.open('enviroTab_pa_train.csv')
    env_data_train = pd.read_csv(env_train_file, sep=';')
    # Remove rows with at least one NA 
    env_data_train = env_data_train.dropna()
    env_data_train['tmpId'] = np.core.defchararray.add(
        [str(el) + '_' for el in env_data_train['patchID']],
        [str(el) for el in env_data_train['dayOfYear']]
    )
    env_test_file = zf.open('enviroTab_pa_test.csv')
    env_data_test = pd.read_csv(env_test_file, sep=';')
    
    ## Environment models
    erz = zf.open('enviro_rf.zip')
    erzf = zipfile.ZipFile(erz)
    files_rf = erzf.infolist()
    models = {}
    for f in tqdm.tqdm(list(filter(lambda x: x.filename.endswith('.pkl'), files_rf))):
        speciesId = f.filename.split('_')[2][:-4]
        if speciesId not in pa_species: continue
        try: models[speciesId] = joblib.load(
            erzf.open(f)
        )
        
        except (ValueError, zipfile.BadZipFile) as e:
            print(f, e)
    
    #list(filter(lambda x: not x.filename.endswith('.pkl'), files_rf))
    # --> mainly pickle files with the trained models
    # Load example pickle model
    #pickled_model_file = erzf.open('enviro_rf/speciesId_1823.pkl')
    #pickled_model = joblib.load(pickled_model_file)

elif model_type == 'spatial':
    ## Spatial models
    srz = zf.open('spatial_rf.zip')
    srzf = srz.infolist()
    srzf = zipfile.ZipFile(srz)
    files_sp = srzf.infolist()
    f=zf.open('maxent_PA_svd/spModelStat_svd.csv')
    #TODO load models

# Merge presence and model input data
if presence_only:
    pass
else:
    ## Training
    pa_train = pa_train[pa_train['tmpId'].isin(env_data_train['tmpId'])]
    pa_train = pa_train.groupby(['tmpId','speciesId']).first().reset_index()
    
    multi_label_pa_train = pa_train[['tmpId','speciesId']]
    multi_label_pa_train.loc[:, 'label'] = 1
    multi_label_pa_train = pd.pivot(multi_label_pa_train,
        index='tmpId',
        columns='speciesId',
        values='label').reset_index().fillna(0)
    
    X = multi_label_pa_train[
        ['tmpId']].merge(
            right=env_data_train, how='left', on='tmpId'
        ).drop(
            ['tmpId', 'dayOfYear', 'patchID', 'lon', 'lat',
             'x_EPSG3035', 'y_EPSG3035','year','dataset'],
            axis=1
        )

    ## Test set
    Xte = env_data_test.drop(
        ['Id', 'dayOfYear', 'patchID', 'lon', 'lat',
         'x_EPSG3035','y_EPSG3035','year','datasetName'], axis=1
    )

# Get species name
def get_occurrence(occurrenceId):
    response = requests.get(
        'https://api.gbif.org/v1/occurrence/'+occurrenceId,
        headers={
            'Accept':'application/json',
            'Content-Type':'application/json'
        }
    )
    return response.json()

speciesId_map_file = '/data/results/kaggle_speciesId_map.csv'
if not os.path.exists(speciesId_map_file):
    speciesIdMap = {}
    for s,grp in presence_data.groupby('speciesId'):
        for occurrenceId in grp.gbifID:
            try:
                speciesIdMap[s] = get_occurence(str(int(occurrenceId)))['species']
                break
            except JSONDecodeError:
                print('Occurence data removed for', occurrenceId)
            except KeyError:
                if s not in speciesIdMap:
                    speciesIdMap[s] = get_occurrence(
                        str(int(occurrenceId))
                    )['scientificName']
                    # TODO track which entries have scientificName
    with open(speciesId_map_file, 'wt') as out:
        for s in speciesIdMap:
            out.write(f"{s},{speciesIdMap[s]}\n")
else:
    with open(speciesId_map_file, 'rt') as f:
        speciesIdMap = dict([l.strip().split(',') for l in f])
        
kaggle_species = pd.DataFrame({'species_name':pd.Series(speciesIdMap)})
if not presence_only:
    kaggle_species = kaggle_species[kaggle_species.index.isin(pa_species)]

name2id = {v:k for k,v in speciesIdMap.items()}
def run_model(speciesId, speciesIdMap, X, multi_label_pa_train, models):
    model = models[speciesId]
    Y = multi_label_pa_train[int(speciesId)]
    probs = model.predict_proba(X.values)[:,1]
    return probs
probs = {
    s:run_model(s, speciesIdMap, X, multi_label_pa_train, models)
    for s in models
}
probs_test = {
    s:models[s].predict_proba(Xte.values)[:,1]
    for s in models
}
del models # clear model memory
max_probs = pd.Series({s:probs[s].max() for s in probs})
max_probs_test = pd.Series({s:probs_test[s].max() for s in probs})
print(spearmanr(max_probs, max_probs_test))
predictions = pd.Series({s:(probs[s]>0.5).sum() for s in probs})
predictions_test = pd.Series({s:(probs_test[s]>0.5).sum() for s in probs})
print(max_probs.describe())
print(multi_label_pa_train.drop('tmpId',axis=1).sum().describe())
# Amount of species with at least 1 location prob > 0.5
print((max_probs>0.5).sum()) #(predictions>0).sum()

# Species interactions
ig = InteractionsGraph()

kaggle_species['ix_r_types'] = kaggle_species.species_name.apply(
    lambda x: ig.run_query(
        f"MATCH (n)-[r]-() WHERE n.name = '{x}' RETURN TYPE(r), COUNT(*)"
    )
)

rtypes = {r:0 for r in ig.relationships}
r_types = pd.DataFrame(list(kaggle_species.ix_r_types.apply(
    lambda x: rtypes|{r['TYPE(r)']:r['COUNT(*)'] for r in x})), index=kaggle_species.index)
r_types['total_r'] = r_types.sum(axis=1)
r_types['species_name'] = kaggle_species.species_name
top20ix = r_types.set_index('species_name').sum().sort_values(ascending=False).index[1:21]
fig, ax = plt.subplots(figsize=(10,5))
sns.boxplot(data=r_types.set_index('species_name')[top20ix], orient='h', ax=ax)
ax.set_xlabel('Interactions / species')
fig.tight_layout()
fig.savefig('/data/results/ix_types_kaggle.png')

species_with_ix = list(r_types[r_types.total_r>0].species_name)
ig.run_query('''MATCH (n) WHERE n.name IN $species_list RETURN COUNT(n)  
     ''', species_list=species_with_ix)

## Dyadic
dyadic_r = ig.run_query('''MATCH (n) WHERE n.name IN $species_list       
       WITH n MATCH (n)-[r]-(m) WHERE m.name IN $species_list
       RETURN TYPE(r), COUNT(*)
''', species_list=species_with_ix)
dyadic_nodes = ig.run_query('''MATCH (n) WHERE n.name IN $species_list       
       WITH n MATCH (n)-[r]-(m) WHERE m.name IN $species_list
       AND m.name <> n.name
       RETURN n.name,TYPE(r),m.name
''', species_list=species_with_ix)
dyadic_nodes = pd.DataFrame(dyadic_nodes)

## Triadic
triadic_r = ig.run_query('''MATCH (n) WHERE n.name IN $species_list
       WITH n MATCH (n)-[r]-(m)-[q]-(o) WHERE m.name IN $species_list         
       AND o.name in $species_list RETURN TYPE(r),TYPE(q),COUNT(*)            
''', species_list=species_with_ix)
triadic_nodes = ig.run_query('''MATCH (n) WHERE n.name IN $species_list
       WITH n MATCH (n)-[r]-(m)-[q]-(o) WHERE m.name IN $species_list         
       AND o.name in $species_list
       RETURN n.name,TYPE(r),m.name,TYPE(q),o.name
''', species_list=species_with_ix)
triadic_nodes = pd.DataFrame(triadic_nodes)
triadic_closed_r = ig.run_query('''MATCH (n) WHERE n.name IN $species_list                                                                      
       WITH n MATCH (n)-[r]-(m)-[q]-(o)-[s]-(n) WHERE m.name IN $species_list  
       AND o.name in $species_list RETURN TYPE(r),TYPE(q),TYPE(s),COUNT(*)     
''', species_list=species_with_ix)
triadic_closed_nodes = ig.run_query('''MATCH (n) WHERE n.name IN $species_list
       WITH n MATCH (n)-[r]-(m)-[q]-(o)-[s]-(n) WHERE m.name IN $species_list  
       AND o.name in $species_list
       RETURN n.name,TYPE(r),m.name,TYPE(q),o.name,TYPE(s)
''', species_list=species_with_ix) 
triadic_closed_nodes = pd.DataFrame(triadic_closed_nodes)

species_kingdom = ig.run_query('''MATCH (n) WHERE n.name IN $species_list
RETURN n.kingdom,COUNT(*)''', species_list=species_with_ix)

# Probability correlations
## Dyadic
model_available = dyadic_nodes.apply(
    lambda x:
        name2id[x['n.name']] in probs
        and name2id[x['m.name']] in probs,
    axis=1
)
dyadic_nodes = dyadic_nodes[model_available].copy()
dyadic_nodes['prob_spears'] = dyadic_nodes.apply(
    lambda x: spearmanr(
        probs[name2id[x['n.name']]],
        probs[name2id[x['m.name']]]),
    axis=1
)
dyadic_nodes['prob_spears_r'] = dyadic_nodes.prob_spears.apply(lambda x: x[0])
print(dyadic_nodes.prob_spears_r.describe())
dyadic_nodes['prob_test_spears'] = dyadic_nodes.apply(
    lambda x: spearmanr(
        probs_test[name2id[x['n.name']]],
        probs_test[name2id[x['m.name']]]),
    axis=1
)
dyadic_nodes['prob_test_spears_r'] = dyadic_nodes.prob_test_spears.apply(
    lambda x: x[0]
)
print(dyadic_nodes.prob_test_spears_r.describe())
print(spearmanr(dyadic_nodes.prob_spears_r, dyadic_nodes.prob_test_spears_r))
dyadic_random = pd.DataFrame({
    'n.name':random.sample(sorted(probs), k=500),
    'm.name':random.sample(sorted(probs), k=500)
})
dyadic_random = dyadic_random[
    dyadic_random['n.name']!=dyadic_random['m.name']
]
dyadic_random['prob_spears'] = dyadic_random.apply(
    lambda x: spearmanr(
        probs[x['n.name']],
        probs[x['m.name']]),
    axis=1
)
dyadic_random['prob_spears_r'] = dyadic_random.prob_spears.apply(lambda x: x[0])
print(dyadic_random.prob_spears_r.describe())

dyadic_nodes['SPECIESSETSTR'] = dyadic_nodes.T.apply(lambda x: str(list(sorted({x['n.name'],x['m.name']}))))
selection=~dyadic_nodes[['SPECIESSETSTR','TYPE(r)']].duplicated()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,5), sharey=True, width_ratios=[8,1])
sns.stripplot(data=dyadic_nodes[selection], y='prob_spears_r', x='TYPE(r)', orient='v', ax=axes[0])
sns.boxplot(data=dyadic_nodes[selection], y='prob_spears_r', x='TYPE(r)', orient='v', ax=axes[0], color='white')#, scale='width')
axes[0].set_ylim((-1,1))
#fig.savefig('/data/results/dyadic_spears_r.png')
#fig, ax = plt.subplots(figsize=(3,5))
sns.stripplot(data=dyadic_random.prob_spears_r, orient='v', ax=axes[1])
sns.boxplot(data=dyadic_random.prob_spears_r, orient='v', ax=axes[1], color='white')#, scale='width')
axes[1].set_xticklabels(['random'])
fig.savefig('/data/results/dyadic_random_spears_r.png')

# dyadic_nodes['correlation_quantile'] = pd.qcut(dyadic_nodes.prob_spears_r, 4)
# fig, ax = plt.subplots(figsize=(10,5))
# sns.stripplot(data=dyadic_nodes[selection], y='corrchecks',
# x='correlation_quantile', orient='v', ax=ax)
# sns.boxplot(data=dyadic_nodes[selection], y='corrchecks',
# x='correlation_quantile', orient='v', ax=ax, color='white')
# fig.savefig('/data/results/dyadic_qcorrchecks.png')
# spearmanr(dyadic_nodes.prob_spears_r, dyadic_nodes.corrchecks)
# pearsonr(dyadic_nodes.prob_spears_r, dyadic_nodes.corrchecks)
# fig2, ax = plt.subplots(figsize=(10,5))
# ax.scatter(dyadic_nodes.prob_spears_r, dyadic_nodes.corrchecks)
# fig2.savefig('/data/results/scatter_dyadic_qcorrchecks.png')

## Triadic
mask_diagonal = np.ones((3,3))
np.fill_diagonal(mask_diagonal, 0)
models_available_t = triadic_nodes.apply(
    lambda x:
        name2id[x['n.name']] in probs
        and name2id[x['m.name']] in probs
        and name2id[x['o.name']] in probs,
    axis=1
)
triadic_nodes_sample = triadic_nodes[
    models_available_t &
    (triadic_nodes['n.name']!=triadic_nodes['m.name']) &
    (triadic_nodes['n.name']!=triadic_nodes['o.name']) &
    (triadic_nodes['m.name']!=triadic_nodes['o.name'])
].copy() #sample(n=1000, random_state=None)
triadic_nodes_sample['prob_spears'] = triadic_nodes_sample.apply(
    lambda x: spearmanr([
        probs[name2id[x['n.name']]],
        probs[name2id[x['m.name']]],
        probs[name2id[x['o.name']]]
        ], axis=1), axis=1
)
triadic_nodes_sample['prob_spears_max_r'] = triadic_nodes_sample.prob_spears.apply(
    lambda x: np.abs(np.multiply(x[0], mask_diagonal)).max()
)
triadic_nodes_sample['prob_spears_mean_r'] = triadic_nodes_sample.prob_spears.apply(
    # Dropping diagonal 1 correlations to calculate mean
    # TODO consider geometric mean
    lambda x: (np.abs(x[0]).sum()-3)/9
)
triadic_nodes_sample['TYPESET'] = triadic_nodes_sample.apply(lambda x: {x['TYPE(r)'], x['TYPE(q)']}, axis=1)
triadic_nodes_top = triadic_nodes_sample[
    triadic_nodes_sample.TYPESET.isin(
        triadic_nodes_sample.TYPESET.value_counts().index[:5])
]
triadic_nodes_top['TYPESETSTR'] = triadic_nodes_top.TYPESET.astype(str)
print(triadic_nodes_sample.prob_spears_mean_r.describe())
### Random triadic model (also for closed triads)
triadic_random = pd.DataFrame({
    'n.name':random.sample(sorted(probs), k=100),
    'm.name':random.sample(sorted(probs), k=100),
    'o.name':random.sample(sorted(probs), k=100)
})
triadic_random = triadic_random[
    (triadic_random['n.name']!=triadic_random['m.name']) &
    (triadic_random['n.name']!=triadic_random['o.name']) &
    (triadic_random['m.name']!=triadic_random['o.name'])
]
triadic_random['prob_spears'] = triadic_random.apply(
    lambda x: spearmanr([
        probs[x['n.name']], probs[x['m.name']], probs[x['o.name']]
    ], axis=1), axis=1
)
triadic_random['prob_spears_mean_r'] = triadic_random.prob_spears.apply(
    lambda x: (np.abs(x[0]).sum()-3)/6
)
print(triadic_random.prob_spears_mean_r.describe())
triadic_nodes_top.TYPESETSTR = triadic_nodes_top.TYPESETSTR.apply(lambda x: x.replace(',',',\n'))

triadic_nodes_top['SPECIESSETSTR'] = triadic_nodes_top.T.apply(lambda x: str(list(sorted({x['n.name'],x['m.name'],x['o.name']}))))
selection=~triadic_nodes_top[['SPECIESSETSTR','TYPESETSTR']].duplicated()
# in theory selection could remove valid typeset variations with same species
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,5), sharey=True, width_ratios=[8,1])
sns.stripplot(data=triadic_nodes_top[selection], y='prob_spears_mean_r', x='TYPESETSTR', orient='v', ax=axes[0])
sns.violinplot(data=triadic_nodes_top[selection], y='prob_spears_mean_r', x='TYPESETSTR', orient='v', scale='width', ax=axes[0], color='white')
#fig.savefig('/data/results/triadic_spears_r.png')
#fig, ax = plt.subplots(figsize=(3,5))
sns.stripplot(data=triadic_random.prob_spears_mean_r, orient='v', ax=axes[1])
sns.violinplot(data=triadic_random.prob_spears_mean_r, orient='v', scale='width', ax=axes[1], color='white')
axes[0].set_ylim((0,1))
axes[1].set_xticklabels(['random'])
fig.tight_layout()
fig.savefig('/data/results/triadic_random_spears_r.png')

## Triadic closed
models_available_tc = triadic_closed_nodes.apply(
    lambda x:
        name2id[x['n.name']] in probs
        and name2id[x['m.name']] in probs
        and name2id[x['o.name']] in probs,
    axis=1
)
triadic_closed_nodes_sample = triadic_closed_nodes[
    models_available_tc &
    (triadic_closed_nodes['n.name']!=triadic_closed_nodes['m.name']) &
    (triadic_closed_nodes['n.name']!=triadic_closed_nodes['o.name']) &
    (triadic_closed_nodes['m.name']!=triadic_closed_nodes['o.name'])
].copy() #sample(n=1000, random_state=None)
triadic_closed_nodes_sample['prob_spears'] = triadic_closed_nodes_sample.apply(
    lambda x: spearmanr([
        probs[name2id[x['n.name']]],
        probs[name2id[x['m.name']]],
        probs[name2id[x['o.name']]]
        ], axis=1), axis=1
)
triadic_closed_nodes_sample['prob_spears_max_r'] = triadic_closed_nodes_sample.prob_spears.apply(
    lambda x: np.abs(np.multiply(x[0], mask_diagonal)).max()
)
triadic_closed_nodes_sample['prob_spears_mean_r'] = triadic_closed_nodes_sample.prob_spears.apply(
    # Dropping diagonal 1 correlations to calculate mean
    # TODO consider geometric mean
    lambda x: (np.abs(x[0]).sum()-3)/6
)
triadic_closed_nodes_sample['train_total_obs'] = triadic_closed_nodes_sample.apply(
    lambda x: multi_label_pa_train[int(name2id[x['n.name']])].sum()+
    multi_label_pa_train[int(name2id[x['m.name']])].sum()+
    multi_label_pa_train[int(name2id[x['o.name']])].sum(), axis=1
)
triadic_closed_nodes_sample['train_shared_obs'] = triadic_closed_nodes_sample.apply(
    lambda x: (
        multi_label_pa_train[int(name2id[x['n.name']])].astype(bool)&
        multi_label_pa_train[int(name2id[x['m.name']])].astype(bool)&
        multi_label_pa_train[int(name2id[x['o.name']])].astype(bool)
    ).sum(), axis=1
)
print(triadic_closed_nodes_sample.prob_spears_mean_r.describe())
print(
    triadic_closed_nodes_sample.groupby(
        ['TYPE(r)','TYPE(q)','TYPE(s)']
    )['prob_spears_mean_r'].describe().sort_values(by='mean')
)

### Make top relationship type selection for plotting
triadic_closed_nodes_sample['TYPESET'] = triadic_closed_nodes_sample.apply(lambda x: {x['TYPE(r)'], x['TYPE(q)'], x['TYPE(s)']}, axis=1)
triadic_closed_nodes_top = triadic_closed_nodes_sample[
    triadic_closed_nodes_sample.TYPESET.isin(
        triadic_closed_nodes_sample.TYPESET.value_counts().index[:5])
]
triadic_closed_nodes_top['TYPESETSTR'] = triadic_closed_nodes_top.TYPESET.astype(str)
triadic_closed_nodes_top['SPECIESSETSTR'] = triadic_closed_nodes_top.T.apply(lambda x: str(list(sorted({x['n.name'],x['m.name'],x['o.name']}))))
selection=~triadic_closed_nodes_top[['SPECIESSETSTR','TYPESETSTR']].duplicated()
# in theory selection could remove valid typeset variations with same species

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,5), sharey=True, width_ratios=[8,1])
sns.stripplot(data=triadic_closed_nodes_top[selection], y='prob_spears_mean_r', x='TYPESETSTR', orient='v', ax=axes[0])
sns.violinplot(data=triadic_closed_nodes_top[selection], y='prob_spears_mean_r', x='TYPESETSTR', orient='v', scale='width', ax=axes[0], color='white')
sns.stripplot(data=triadic_random.prob_spears_mean_r, orient='v', ax=axes[1])
sns.violinplot(data=triadic_random.prob_spears_mean_r, orient='v', scale='width', ax=axes[1], color='white')
axes[0].set_ylim((0,1))
axes[1].set_xticklabels(['random'])
fig.savefig('/data/results/triadic_closed_random_spears_r.png')

## Get embeddings for interacting species
#ig.run_query('''MATCH (n:species)-[r]-()
#  SET r.occurrences = toInteger(r.occurrences) RETURN COUNT(r)
#''')
ea = EcoAnalysis(ig)
ea.create_projection(
    'kaggle_projection',
    ['species'],
    {
        r:{'properties':['occurrences']}
        for r in dyadic_nodes['TYPE(r)'].value_counts().index
    }, force=True
)
ea.create_embedding('kaggle_projection', embeddingDimension=10, force=True)
embeddings = ea.get_embeddings(
    'kaggle_projection',
    species_list=species_with_ix, plot=False
)
ea.drop_projection('kaggle_projection')
embeddings = embeddings.groupby('name').apply(
    lambda g:  np.mean(np.stack(g.embedding), axis=0)
)
embeddings = pd.DataFrame(
    embeddings.to_list(), index=embeddings.index
)
if test_randomised_embeddings:
    for c in embeddings.columns:
        embeddings[c] = list(embeddings[c].sample(frac=1,ignore_index=True))
        
# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn import metrics

# Train model with only embeddings and species own prob prediction
# Xemb = []
# yemb = []
# for species in embeddings.index:
#     e = embeddings.loc[species].to_list()
#     si = name2id[species]
#     if si not in probs: continue
#     Xemb += [[p]+e for p in probs[si]]
#     yemb += multi_label_pa_train[int(si)].to_list()
# Xemb = np.array(Xemb)
# yemb = np.array(yemb)

# Train model with embeddings, species own prob prediction, PCA of other predictions
probs = pd.DataFrame(probs)
pca = PCA(n_components=10)
if pca_fit_on_subset:
    probs_subset = probs.sample(frac=0.5)
    pca.fit(probs_subset)
else:
    pca.fit(probs)
print(pca.explained_variance_ratio_)
#print(pca.singular_values_)
probs_pca = pca.transform(probs)
probs_pca = pd.DataFrame(probs_pca)
Xemb = []
yemb = []
added_subset = False
for species in embeddings.index:
    e = embeddings.loc[species].to_list()
    si = name2id[species]
    if si not in probs: continue
    probs_pca['species_prob'] = probs[si]
    for i,ed in enumerate(e):
        probs_pca[f"emb{i}"] = ed
    if not added_subset and pca_fit_on_subset:
        probs_pca['used2fit'] = probs_pca.index.isin(probs_subset.index)
        added_subset = True
    probs_pca['species'] = species
    Xemb.append(probs_pca.values)
    yemb += multi_label_pa_train[int(si)].to_list()
Xemb = np.concatenate(Xemb)
yemb = np.array(yemb)
Xemb, yemb = resample( # Reduce set to avoid mem killing
    Xemb, yemb, n_samples=500000, replace=False, random_state=33
)

# Train/test prep
Xemb_train, Xemb_test, yemb_train, yemb_test = train_test_split(
    Xemb, yemb, test_size=0.5, stratify=yemb, random_state=42
)
# Reduce test set to avoid memory issue
#Xemb_test, yemb_test = resample(
#    Xemb_test, yemb_test, n_samples=100000, replace=False, random_state=33
#)
rus = RandomUnderSampler(
    sampling_strategy=0.5, random_state=0
)
Xemb_resampled, yemb_resampled = rus.fit_resample(
    Xemb_train, yemb_train
)

## Include total of non-0 embedding dimensions
# TODO add directly when creating Xemb
Xemb_resampled = np.concatenate([
    Xemb_resampled,
    (Xemb_resampled[:,11:21]!=0).sum(axis=1).reshape(-1, 1)
], axis=1)
Xemb_test = np.concatenate([
    Xemb_test,
    (Xemb_test[:,11:21]!=0).sum(axis=1).reshape(-1, 1)
], axis=1)

# Species traits
cattdf = pd.read_csv(
    '/data/species_traits/Try2023811103613TRY_Categorical_Traits_Lookup_Table_2012_03_17_TestRelease/TRY_Categorical_Traits_Lookup_Table_2012_03_17_TestRelease.csv',
    sep=';', low_memory=False
)
cattdf = cattdf[
    cattdf.AccSpeciesName.isin(
        species_with_ix
        #kaggle_species.species_name
    )
].set_index('AccSpeciesName')
trait_selectors = {
    pgf:[
        cattdf.loc[s].PlantGrowthForm==pgf
        if s in cattdf.index else False
        for s in Xemb_test[:,22]
    ] for pgf in cattdf.PlantGrowthForm.value_counts().index
}
trait_selectors = {
    f"{ts} ({sum(trait_selectors[ts])})":trait_selectors[ts]
    for ts in trait_selectors
}
    
# Fit and validate ML models
pca_trained_selector = Xemb_test[:,21].astype(bool)
spec_embed_pyth = (Xemb_test[:,11:21]**2).sum(axis=1)**.5
spec_embed_selector = spec_embed_pyth>.9 #Xemb_test[:,22]>9
for input_type, Xemb_select, Xemb_test_select in zip(
        ('pca_s_emb', 's_emb', 'pca_s', 'spred', 'emb', 'pca', 'sumemb'),
        (Xemb_resampled[:,:21], Xemb_resampled[:,10:21],
         Xemb_resampled[:,:11],
         Xemb_resampled[:,10].reshape(-1, 1),
         Xemb_resampled[:,11:21],
         Xemb_resampled[:,:10],
         Xemb_resampled[:,
                        (23 if pca_fit_on_subset else 22)
                        ].reshape(-1, 1)),
        (Xemb_test[:,:21], Xemb_test[:,10:21], Xemb_test[:,:11],
         Xemb_test[:,10].reshape(-1, 1),
         Xemb_test[:,11:21], Xemb_test[:,:10],
         Xemb_test[:,
                   (23 if pca_fit_on_subset else 22)
                   ].reshape(-1, 1))
):
    print(input_type)
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(Xemb_select, yemb_resampled)
    score = clf.score(Xemb_test_select, yemb_test)
    print('Score', score)
    probs_emb = clf.predict_proba(Xemb_select)[:,1]
    probs_emb_test = clf.predict_proba(Xemb_test_select)[:,1]
    preds_emb_test = clf.predict(Xemb_test_select)
    print(
        metrics.confusion_matrix(yemb_test, preds_emb_test),
        '\nPCA trained\n', metrics.confusion_matrix(
            yemb_test[pca_trained_selector],
            preds_emb_test[pca_trained_selector]
        ),
        '\nNot PCA trained\n', metrics.confusion_matrix(
            yemb_test[~pca_trained_selector],
            preds_emb_test[~pca_trained_selector]
        ),
        '\nHigh spec embed\n', metrics.confusion_matrix(
            yemb_test[spec_embed_selector],
            preds_emb_test[spec_embed_selector]
        ),
        '\nLow spec embed\n', metrics.confusion_matrix(
            yemb_test[~spec_embed_selector],
            preds_emb_test[~spec_embed_selector]
        )
    )
    fig, axes = plt.subplots(2, 7, figsize=(20,10))
    for pgfi, pgf in enumerate(trait_selectors):
        print(
            pgf, '\n',
            metrics.confusion_matrix(
                yemb_test[trait_selectors[pgf]],
                preds_emb_test[trait_selectors[pgf]]
            )
        )
        metrics.RocCurveDisplay.from_predictions(
            yemb_test[trait_selectors[pgf]],
            probs_emb_test[trait_selectors[pgf]], ax=axes[0,pgfi]
        )
        axes[0,pgfi].set_title(pgf)
        metrics.PrecisionRecallDisplay.from_predictions(
            yemb_test[trait_selectors[pgf]],
            probs_emb_test[trait_selectors[pgf]], ax=axes[1,pgfi]
        )
    fig.savefig(f"/data/results/{input_type}_pred_roc_on_traits.png")
    #print(
    #    pd.Series(probs_emb_test[yemb_test==1]).describe(),
    #    pd.Series(probs_emb_test[yemb_test==0]).describe()
    #)
    #fpr, tpr, thresholds = metrics.roc_curve(yemb_test, probs_emb_test)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,10))
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    #ax.plot(fpr, tpr)
    metrics.RocCurveDisplay.from_predictions(yemb_test, probs_emb_test, ax=ax1)
    #metrics.RocCurveDisplay.from_predictions(yemb_resampled, probs_emb, ax=ax)
    ax1.set_xlabel('False positive rate')
    ax1.set_ylabel('True positive rate')
    metrics.PrecisionRecallDisplay.from_predictions(yemb_test, probs_emb_test, ax=ax2)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    # Confusion spread plots
    tp = (yemb_test==1)&(preds_emb_test==1)
    tn = (yemb_test==0)&(preds_emb_test==0)
    fp = (yemb_test==0)&(preds_emb_test==1)
    fn = (yemb_test==1)&(preds_emb_test==0)
    for sel,label in zip((tn,fn,fp,tp),('tn','fn','fp','tp')):
        ## PCA dim1&2 confusion
        ax3.scatter(Xemb_test[sel, 0],Xemb_test[sel, 1], label=label, alpha=.1)
        ## Species embedding  dim1&2 confusion
        ax4.scatter(Xemb_test[sel, 11],Xemb_test[sel, 12], label=label, alpha=.1)
    ax3.set_title('PCA confusion')
    lgd = ax3.legend()
    for lh in lgd.legend_handles: lh.set_alpha(1)
    ax4.set_title('Species embedding confusion')
    lgd = ax4.legend()
    for lh in lgd.legend_handles: lh.set_alpha(1)
    fig.suptitle(f"{input_type} performance - score: {score}")
    fig.savefig(f"/data/results/{input_type}_pred_roc.png")

# Train model with embeddings, species own prob prediction, all other predictions
# Xemb = []
# for species in embeddings.index:
#     e = embeddings.loc[species].to_list()
#     si = name2id[species]
#     if si not in probs: continue
#     Xemb += [
#         [p]+e+
#         [probs[osi][i] for osi in probs] for i,p in enumerate(probs[si])
#     ]
# Xemb = np.array(Xemb)

# Train model with predictions multiplied with embeddings

# Species features
## https://www.ebi.ac.uk/ols/ontologies/oba
## https://pubmed.ncbi.nlm.nih.gov/36747660/
## https://pubmed.ncbi.nlm.nih.gov/19779746/
from guardgraph.utils import download_file
from owlready2 import get_ontology
download_file(
    'http://purl.obolibrary.org/obo/oba.owl',
    '/data'
)
onto = get_ontology("file:///data/oba.owl").load()

## Species traits dataset
### Open traits interesting sets
#### PhenObs, UCIMLR-Iris, AusTraits, GlobTherm, BETYdb, BROT, Compadre, Database of Plant Heat Tolerances, eFLOWER, Elton Traits, Global Biotic Interactions, Kubitzki et al, Santi et al, 2013, TRY - Global Plant Trait Database
### https://opentraits.org/datasets/PhenObs
### data @ https://idata.idiv.de/ddm/Data/ShowData/3535?version=38
### cite @ https://doi.org/10.25829/idiv.3535-6j8cmx
#traits = pd.read_csv(
#    '/data/species_traits/3535_38_processeddata_PhenObs_2020(1).csv',
#    sep=';'
#)                                    
#trait_species = set(traits.Species.value_counts().index)

### Try db
#### For citation see zip
#%pip install openpyxl
# read_excel gave seg fault so manually exported it to csv
#### Interesting columns: PhylogeneticGroup, PlantGrowthForm, LeafType, LeafPhenology, PhotosyntheticPathway, Woodiness, LeafCompoundness
# cattdf = pd.read_csv(
#     '/data/species_traits/Try2023811103613TRY_Categorical_Traits_Lookup_Table_2012_03_17_TestRelease/TRY_Categorical_Traits_Lookup_Table_2012_03_17_TestRelease.csv',
#     sep=';'
# )
# cattdf = cattdf[
#     cattdf.AccSpeciesName.isin(
#         species_with_ix
#         #kaggle_species.species_name
#     )
# ].set_index('AccSpeciesName')
# for pgf in cattdf.PlantGrowthForm.value_counts().index:
#     selector = [
#         cattdf.loc[s].PlantGrowthForm==pgf
#         if s in cattdf.index else False
#         for s in Xemb_test[:,22]
#     ]  
#     print(
#         pgf, '\n',
#         metrics.confusion_matrix(yemb_test[selector], preds_emb_test[selector])
#     )

# GeoPlotting Interactions
from guardgraph.geo import plot_countries
fig, ax = plot_countries([
    ('France métropolitaine', 3),                                            
    ('England', 4),                                                          
    ('Cymru / Wales', 4),                                                    
    ('Alba / Scotland', 4)                                                   
])

geo_ix = pa_train.groupby(
    ['lon','lat']
).apply(
    lambda grp: len(set(grp.speciesId.astype('str'))&set(kaggle_species.index))/
    len(grp.speciesId.unique())
).reset_index().rename({0:'count_ix'},axis=1)
ax.scatter(geo_ix.lon, geo_ix.lat,  c=geo_ix.count_ix, alpha=.7)

