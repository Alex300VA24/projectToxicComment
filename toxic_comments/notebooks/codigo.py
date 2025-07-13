from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# 🟢 Paso 0: MUÉSTREO
train_sampled = train.sample(50000, random_state=42)

# 🟢 Paso 1: Transformar lista de listas en matriz booleana
te = TransactionEncoder()
te_ary = te.fit(train_sampled['transacoes']).transform(train_sampled['transacoes'])

df_trans = pd.DataFrame(te_ary, columns=te.columns_)

print("Formato de la matriz transaccional:", df_trans.shape)
display(df_trans.head())

# 🟢 Paso 2: Obtener itemsets frecuentes
frequent_itemsets = apriori(
    df_trans,
    min_support=0.5,
    use_colnames=True
)

print("Itemsets frequentes encontrados:", len(frequent_itemsets))
display(frequent_itemsets.head())

# 🟢 Paso 3: Generar reglas de asociación
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.5
)

# 🟢 Paso 4: Filtrar reglas con lift >1
rules = rules[rules["lift"] > 1]

print("Reglas generadas:", len(rules))
display(rules.head())




# ----------------
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from collections import Counter
from scipy.sparse import csr_matrix

# 🟢 Paso 0: MUÉSTREO (reduce cantidad de filas)
train_sampled = train.sample(10000, random_state=42)   # Solo 10,000 filas

# 🟢 Paso 1: Filtrado de vocabulario (reduce cantidad de palabras únicas)

# 1.1 Contar todas las palabras en la muestra
all_words = []
for tx in train_sampled['transacoes']:
    all_words.extend([item for item in tx if item.startswith("palavra=")])

word_counts = Counter(all_words)

# 1.2 Mantener solo las TOP N palabras más frecuentes
N = 2000    # Máximo 2,000 palabras
most_common_words = set([w for w, c in word_counts.most_common(N)])

# 1.3 Función de filtrado
def filtrar_transaccion(tx):
    return [item for item in tx if not item.startswith("palavra=") or item in most_common_words]

# 1.4 Aplicar filtrado
train_sampled['transacciones_filtradas'] = train_sampled['transacoes'].apply(filtrar_transaccion)

# 🟢 Paso 2: Transformar lista de listas en matriz booleana dispersa
te = TransactionEncoder()
te_ary = te.fit(train_sampled['transacciones_filtradas']).transform(train_sampled['transacciones_filtradas'])

# Crear DataFrame disperso
df_trans_sparse = pd.DataFrame(te_ary, columns=te.columns_)

print("Formato de la matriz transaccional (sparse):", df_trans_sparse.shape)
display(df_trans_sparse.head())

# 🟢 Paso 3: Obtener itemsets frecuentes
frequent_itemsets = apriori(
    df_trans_sparse,
    min_support=0.5,   # 1% de todas las transacciones
    use_colnames=True
)

print("Itemsets frecuentes encontrados:", len(frequent_itemsets))
display(frequent_itemsets.head())

# 🟢 Paso 4: Generar reglas de asociación
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.7    # Confianza mínima 70%
)

# 🟢 Paso 5: Filtrar reglas con lift > 1.2
rules = rules[rules["lift"] > 1.2]

print("Reglas generadas:", len(rules))
display(rules.sort_values(by="lift", ascending=False).head(10))   # Mostrar las top 10
