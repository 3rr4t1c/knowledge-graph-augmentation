import pandas as pd

# Mostra delle statistiche per un DataFrame passato per parametro.
def custom_stats(df, max_rows=6, col_names=None):
    nrows, ncols = df.shape 
    
    print(f'\n >> DataFrame rows: {nrows}, DataFrame columns: {ncols}\n')
    
    target_columns = list(df.columns)
    if col_names:
        target_columns = col_names
        
    for col_name in target_columns:
        col_nunique = df[col_name].nunique()
        col_dtype = df[col_name].dtype
        col_nan = df[col_name].isnull().sum()
        print('************************')
        print(f'Column: "{col_name}" ', end='') 
        print(f'| Uniques: {col_nunique} in {nrows} ', end='') 
        print(f'| Missing: {col_nan} ({int(col_nan / nrows * 100)}%) ', end='') 
        print(f'| Type: {col_dtype}')
        print('\nClasses: \n')
        vcount = df.groupby(col_name).size().sort_values(ascending=False).reset_index()
        vcount['Perc'] = (vcount.iloc[:,1] / nrows * 100).round(3).astype(str) + '%'
        print(vcount.to_string(header=False, index=False, max_rows=max_rows), '\n')


# Crea l'insieme di train e di test, bilanciando l'insieme di test
def custom_train_test_split(df, target_feature, test_split=0.2):
    
    nsample_per_class = int(df.shape[0] * test_split / df[target_feature].nunique())  
    
    # Per versioni >= 1.1.x
    #test_set_df = df.groupby(target_feature).sample(n=nsample_per_class, random_state=1)
    
    test_set_df = df.groupby(target_feature, group_keys=False).apply(pd.DataFrame.sample, n=nsample_per_class)
    
    train_set_df = pd.concat([df,test_set_df]).drop_duplicates(keep=False)

    return train_set_df, test_set_df

 


