# Toxic Comment Classification

Este projeto treina um modelo BERT para detectar comentários tóxicos.  
Dataset de exemplo: [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge)

## Requisitos

- Python 3.8+
- Git (opcional)
- Ambiente virtual (recomendado)

### Criar ambiente virtual

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### Instalar dependências
#### Após ativar o ambiente virtual, instale os pacotes com:
```bash
pip install -r requirements.txt
```
#### Se modificar as dependências, atualize o arquivo requirements.txt com:
```bash
pip freeze > requirements.txt
```
### Usar
Execute o notebook principal localizado em:
```bash
notebooks/toxic_classification.ipynb
```
## Estrutura do Projeto
📦 projeto/
 ┣ 📂 toxic_comments/
 ┃ ┣ 📜 my_topic_model
 ┃ ┣ 📜 toxic_classification.ipynb
 ┣ 📂 data/
 ┃ ┣ 📜 train.csv
 ┃ ┣ 📜 test.csv
 ┃ ┣ 📜 test_labels.csv
 ┃ ┣ 📜 sample_submission.csv
 ┣ 📂 notebooks/
 ┃ ┣ 📜 toxic_classification.ipynb
 ┣ 📜 requirements.txt
 ┣ 📜 .gitignore
 ┣ 📜 README.md
