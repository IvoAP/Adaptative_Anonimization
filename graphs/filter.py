
# Retorna um dicionário com o melhor desempenho de cada modelo na seguinte estrutura:
# {modelo_1: {acurácia, precisão, recall, f1_score},
#  modelo_2: {acurácia, precisão, recall, f1_score},
#  ...modelo_n: {acurácia, precisão, recall, f1_score}}

def filter_best_results(database, models: list, annonymized: bool):
    results = {}

    for model in models:

        if annonymized:
            db = database[
                (database['model'] == model) & 
                (database['anonymized_train']) & 
                (database['anonymized_test'])
            ]
        
        else:
            db = database[
                (database['model'] == model) & 
                (database['anonymized_train'] == False) & 
                (database['anonymized_test'] == False)
            ]

        highest_accuracy = 0
        highest_precision = 0
        highest_recall = 0
        highest_f1score = 0

        for index, line in db.iterrows():
            if line['accuracy'] > highest_accuracy:
                highest_accuracy = line['accuracy']
            
            if line['precision'] > highest_precision:
                highest_precision = line['precision']
            
            if line['recall'] > highest_recall:
                highest_recall = line['recall']
            
            if line['f1_score'] > highest_f1score:
                highest_f1score = line['f1_score']
        
        results[model] = {
            "highest_accuracy": highest_accuracy,
            "highest_precision":  highest_precision,
            "highest_recall": highest_recall,
            "highest_f1score": highest_f1score
            }
        

    return results


# Retorna, para dada métrica, uma lista na seguinte estrutura:
# [
#   [melhores resultados {metrica} heart anon.], [melhores resultados {metrica} heart nanon.],
#   [melhores resultados {metrica} mgm anon.], [melhores resultados {metrica} mgm nanon.]
# ]

def extract_metric(datasets, metric_name):
    return[
        [model[metric_name] for model in dataset.values()]
        for dataset in datasets
    ]