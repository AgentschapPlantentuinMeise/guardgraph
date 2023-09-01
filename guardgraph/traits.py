import zipfile
trytraits = zipfile.ZipFile('/data/species_traits/28426_22082023125129.zip')
trydata = trytraits.open('28426.txt')
traits = pd.read_csv('/data/species_traits/28426_22082023125129/28426.txt',sep='\t', encoding='latin-1')                                                             
traits.pop(traits.columns[-1]);
traits
traits.StdValueStr.value_counts()
traits.columns
traits.pop('LastName');
traits.pop('FirstName');
traits.Comment
traits.pop('Comment');
traits.Reference
traits.pop('Reference');
traits
traits.columns
traits.loc[0]
species_traits = traits[['SpeciesName','Dataset','TraitName','OriglName']]
species_traits
species_traits.groupby(['SpeciesName','TraitName']).count()
st = species_traits.groupby(['SpeciesName','TraitName'])['OriglName'].agg(set)     
st
traits.columns
species_traits = traits[['SpeciesName','Dataset','TraitName','ValueKindName']]     
st = species_traits.groupby(['SpeciesName','TraitName'])['ValueKindName'].agg(set) 
st
traits.loc[0]
traits.loc[1]
traits.loc[2]
species_traits = traits[['SpeciesName','Dataset','TraitName','OrigValueStr']]      
st = species_traits.groupby(['SpeciesName','TraitName'])['OrigValueStr'].agg(set)  
st
st.reset_index()
st.index
st.unstack()
spectraits = st.unstack().T
spectraits
spectraits.isna()
spectraits.isna().sum()
spectraits.isna().sum().sort_values()
spectraits.isna().sum(axis=1).sort_values()
traits.columns
species_traits = traits[['AccSpeciesName','Dataset','TraitName','ValueKindName']].groupby(['AccSpeciesName','TraitName'])['OrigValueStr'].agg(set)                   
species_traits = traits[['AccSpeciesName','Dataset','TraitName','OrigValueStr']].groupby(['AccSpeciesName','TraitName'])['OrigValueStr'].agg(set)                    
species_traits.unstack()
species_traits.unstack().isna().sum()
species_traits.unstack().isna().sum().sort_values()
species_traits.unstack().isna().sum().sort_values().head(20)
