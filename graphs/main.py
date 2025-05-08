import pandas
import matplotlib.pyplot as plt
from filter import filter_best_results, extract_metric

# Modelos utilizados para os gróficos
modelos = ["KNN", "Random Forest", "GaussianNB", "Mutilayer Perceptron", "AdaBoost", "Logistic Regression"]

# Leitura e armazenamento dos resultados
heart = pandas.read_csv("../results/heart.csv")
mgm = pandas.read_csv("../results/mgm.csv")

# Filtragem pelos melhores resultados por modelo
heartVV = filter_best_results(heart, modelos, True)
heartFF = filter_best_results(heart, modelos, False)
mgmVV = filter_best_results(mgm, modelos, True)
mgmFF = filter_best_results(mgm, modelos, False)

# Armazenamento dos resultados filtrados em lista
datasets_result = [heartVV, heartFF, mgmVV, mgmFF]

# Organização dos dados por dada métrica
highest_accuracy = extract_metric(datasets_result, 'highest_accuracy')
highest_precision = extract_metric(datasets_result, 'highest_precision')
highest_recall = extract_metric(datasets_result, 'highest_recall')
highest_f1score = extract_metric(datasets_result, 'highest_f1score')

# Armazenamento dos melhores resultados para cada modelo em cada métrica em uma lista
values = [highest_accuracy, highest_precision, highest_recall, highest_f1score]

# Configuração do gráfico
cores = ['#E1A36F', '#DEC484', '#E2D8A5', '#6F9F9C', '#577E89', '#A3B9A5']
fig, axs = plt.subplots(4, 4, figsize = (16, 12))
axs = axs.flatten()

# Construção do gráfico
for row in range(4):
    for col in range(4):
        idx = row * 4 + col

        axs[idx].bar(modelos, values[row][col], color = cores)
        axs[idx].set_ylim(0, 1)
        axs[idx].set_xticklabels([])
        axs[idx].tick_params(axis = 'x', labelsize = 4)
        axs[idx].tick_params(axis = 'y', labelsize = 14)

        if col == 0:
            axs[idx].set_ylabel(['Accuracy','Precision','Recall','F1'][row],
                                rotation = 0,
                                ha = 'right',
                                va = 'center',
                                fontsize = 17,
                                labelpad = 15)
        if row == 0:
            axs[idx].set_title(['Heart Anonymized','Heart Not Anonymized','MGM Anonymized','MGM Not Anonymized'][col],
                               fontsize = 17)


plt.tight_layout(pad=2.0, w_pad=2.0, h_pad=3.5)
plt.show()