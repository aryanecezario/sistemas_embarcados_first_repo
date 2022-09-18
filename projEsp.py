from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from keras import layers
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Redimensionará as imagens para este tamanho (o colab pode não possuir memória
# suficiente para imagens maiores do que 100x100)
tam = 100

# Opera as imagens para o treinamento
training_gen = ImageDataGenerator(rescale = 1./255,
                                  rotation_range = 7,
                                  horizontal_flip = True,
                                  shear_range = 0.2,
                                  height_shift_range = 0.07,
                                  zoom_range = 0.2)

# Realiza a leitura do dataset de treino
# base_treinamento = training_gen.flow_from_directory('/home/aryanecezario/Documentos/teste/histograma/train',
#                                                            target_size = (tam,tam),                                                           
#                                                            batch_size=150, 
#                                                            class_mode = 'binary')
# Normaliza os valores
gerador_teste = ImageDataGenerator(rescale = 1./255)

base_treinamento = training_gen.flow_from_directory('/home/aryanecezario/Documentos/teste/png/training_set',
                                                           target_size = (tam,tam),                                                           
                                                           batch_size=30, 
                                                           class_mode = 'binary')

# Realiza a leitura do dataset de validação
# base_validacao = gerador_teste.flow_from_directory('/home/aryanecezario/Documentos/teste/histograma/valid',
#                                                            target_size = (tam,tam),  
#                                                             batch_size=150,                                                          
#                                                            class_mode = 'binary')

base_validacao = gerador_teste.flow_from_directory('/home/aryanecezario/Documentos/teste/png/valid_set/valid_set',
                                                           target_size = (tam,tam),  
                                                            batch_size=30,                                                          
                                                           class_mode = 'binary')

# Realiza a definição da CNN
input_shape = (tam, tam, 3)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(64, kernel_size=(9, 9), activation="relu"),
        layers.MaxPooling2D(pool_size=(5, 5)), 
        layers.Conv2D(64, kernel_size=(5, 5), activation="relu"),
        layers.MaxPooling2D(pool_size=(3, 3)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model.summary()

plot_model(model, '/home/aryanecezario/Documentos/teste/my-CNNmodel.png', show_shapes=True)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

batch_size = 50
epochs = 50

hist = model.fit(base_treinamento, validation_data = base_validacao, batch_size=batch_size, epochs=epochs)

# Testa com o dataset de validação
acc = model.evaluate(base_validacao, steps=len(base_validacao), verbose=0)

# Gráfico de Acurácia por Época
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Acurácia do Modelo')
plt.ylabel('Acurácia')
plt.xlabel('Época')
plt.legend(['treino', 'validação'], loc='upper left')
plt.show()
# plt.savefig('/home/aryanecezario/Documentos/teste/AccAdam.png', format='png')

# Gráfico de Loss por Época
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss do Modelo')
plt.ylabel('Loss')
plt.xlabel('Época')
plt.legend(['treino', 'validação'], loc='upper left')
plt.show()
#plt.savefig('/home/aryanecezario/Documentos/teste/LossAdam.png', format='png')

# Imprime o Loss e a Acurácia final do dataset de validação
loss = acc[0]
accuracy = acc[1]
print("Loss = %.3f" % loss)
print("Acurácia = %.3f" % (accuracy * 100.0) + "%")

# Realiza a leitura do dataset de teste
# base_teste = gerador_teste.flow_from_directory('/home/aryanecezario/Documentos/teste/histograma/test',
#                                                            target_size = (tam,tam),  
#                                                             batch_size=150,                                                         
#                                                            class_mode = 'binary')

# Realiza a leitura do dataset de teste
base_teste = gerador_teste.flow_from_directory('/home/aryanecezario/Documentos/teste/png/test_set',
                                                           target_size = (tam,tam),  
                                                            batch_size=30,                                                         
                                                           class_mode = 'binary')

# Define o Batch Size do ImageDataGenerator do dataset de teste
base_teste.batch_size = len(base_teste.classes)

# Faz as predições
images, labels = next(base_teste)
y_pred = model.predict(images, verbose=0)
predictions = (y_pred > 0.5).astype(int)

# Benign->Benign
true_positive = 0

# Benign->Malignant
false_positive = 0

# Malignant->Malignant
true_negative = 0

# Malignant->Benign
false_negative = 0

for i in range(labels.shape[0]):
    if labels[i].astype(int) == 0:
        if labels[i].astype(int) == predictions[i,0].astype(int):
            true_positive = true_positive + 1
        else:
            false_positive = false_positive + 1
    else:
        if labels[i].astype(int) == predictions[i,0].astype(int):
            true_negative = true_negative + 1
        else:
            false_negative = false_negative + 1

print("         Benign | Malignant")
print("Benign    [ " + str(true_positive) + "   ,   " + str(false_positive) + "]")
print("Malignant [ " + str(false_negative) + "   ,   " + str(true_negative) + "]")

# Imprime a acurácia a partir da matriz de confusão
true_accuracy = (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative)
print('Acurácia no dataset de teste: %.3f' % (true_accuracy * 100.0) + '%')

from sklearn.metrics import confusion_matrix
import seaborn as sns

# notice the threshold
def plot_cm(labels: np.ndarray, predictions: np.ndarray, p: float=0.5):
    cm = confusion_matrix(labels, predictions > p)
    #cm = [[490, 254],[74, 1555]]
    # you can normalize the confusion matrix

    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Matriz de Confusão '.format(p))
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predição')

    print('Nódulos Benignos Identificados como Benignos (True Negatives): ', cm[0][0])
    print('Nódulos Benignos Identificados Como Malignos (False Positives): ', cm[0][1])
    print('Nódulos Malignos Identificados Como Benignos (False Negatives): ', cm[1][0])
    print('Nódulos Malignos Identificados Como Malignos (True Positives): ', cm[1][1])

plot_cm(1, predictions)
plt.savefig('/home/aryanecezario/Documentos/teste/cmAdam.png', format='png')