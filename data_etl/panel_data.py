import pickle
import unidecode
import textdistance
import numpy as np
#Read the pickle file
with open(r'data/all_data.pkl', 'rb') as handle:
    symbols = pickle.load(handle)


columns_a = list(symbols.values())[0].columns.tolist()
columns_b = list(symbols.values())[1].columns.tolist()

#Create the longest common substring similarity distance matrix
def create_lcsstr_matrix(columns_a,columns_b):
    #Step 1: Remove accents and makes everything upper case
    columns_a = [x.upper() for x in columns_a]
    columns_a = [unidecode.unidecode(x) for x in columns_a]
    columns_a = [x.replace(" ", "") for x in columns_a]
    columns_b = [x.upper() for x in columns_b]
    columns_b = [unidecode.unidecode(x) for x in columns_b]
    columns_b = [x.replace(" ", "") for x in columns_b]
    #Step 2 find the levenshtein distance matrix
    dist_mat = np.zeros((len(columns_a),len(columns_b)))
    for row in range(len(columns_a)):
        for col in range(len(columns_b)):
            dist_mat[row,col] = textdistance.lcsstr.distance(columns_a[row],columns_b[col])

    return dist_mat

dist_mat = create_lcsstr_matrix(columns_a,columns_b)

result = np.argmin(dist_mat, axis =1)
print(result)


ipos = 8
txt1 = columns_a[ipos]
print(txt1)
txt2 = columns_b[result[ipos]]
print(txt2)
res1 = textdistance.lcsstr.distance(columns_a[ipos],columns_b[result[ipos]])
print(res1)
res2 = dist_mat[ipos,result[ipos]]
print(res2)



