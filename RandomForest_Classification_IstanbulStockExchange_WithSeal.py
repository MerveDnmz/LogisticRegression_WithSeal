import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from ucimlrepo import fetch_ucirepo
import time
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('/Users/mervedonmez/Documents/VSProjects/SealExample3/roc_curve.png')  # ROC eğrisini doğru dizine kaydet
    plt.close()

def add_gaussian_noise(data, noise_level=0.01):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def classification_problem():
    # Veri setini ucimlrepo ile indir
    istanbul_stock_exchange = fetch_ucirepo(id=247)
    data_classification = pd.DataFrame(istanbul_stock_exchange.data.features)
    
    # Çift kolonları ve gereksiz kolonları temizle
    data_classification = data_classification.loc[:, ~data_classification.columns.duplicated()]  # Çift kolonları temizle
    target_column = 'ISE'
    labels_classification = data_classification[target_column]
    labels_classification = (labels_classification > 0).astype(int).to_numpy().ravel()
    data_classification = data_classification.drop(columns=['date', target_column])  # 'date' ve hedef kolonunu çıkar
    
    # Orijinal sütun adlarını kaydet
    original_columns = data_classification.columns
    
    # SMOTE ile veri setini genişlet
    smote = SMOTE(random_state=42, k_neighbors=10)
    data_classification, labels_classification = smote.fit_resample(data_classification, labels_classification)
    
    # Gaussian Noise ile veri setini genişletme
    augmented_data = add_gaussian_noise(data_classification)
    data_classification = np.vstack([data_classification, augmented_data])
    labels_classification = np.hstack([labels_classification, labels_classification])
    
    # NumPy dizisini tekrar Pandas DataFrame'e dönüştür
    data_classification = pd.DataFrame(data_classification, columns=original_columns)
    
    print(f"Yeni veri seti boyutu: {data_classification.shape}")
    print(f"Yeni etiket boyutu: {labels_classification.shape}")
    print(f"data_classification sütun sayısı: {data_classification.shape[1]}")
    print(f"Sütun adları: {data_classification.columns}")
    
    # Random Forest sınıflandırıcı
    rf_classifier = RandomForestClassifier(random_state=42, class_weight='balanced')
    
    # Hiperparametre optimizasyonu için genişletilmiş parametre ızgarası
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8]
    }
    
    # GridSearchCV ile hiperparametre optimizasyonu
    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(data_classification, labels_classification)
    
    # En iyi parametreler ve en iyi doğruluk
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print(f"Best Parameters: {best_params}")
    print(f"Best Cross-Validated Accuracy: {best_score}")
    
    # En iyi model ile yeniden eğitim ve değerlendirme
    best_rf_classifier = grid_search.best_estimator_
    
    # K-Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    confusion_matrices = []
    all_predictions = []
    all_probabilities = []
    start_time = time.time()

    for train_index, test_index in kf.split(data_classification):
        train_features, test_features = data_classification.iloc[train_index].values, data_classification.iloc[test_index].values
        train_labels, test_labels = labels_classification[train_index], labels_classification[test_index]
        
        best_rf_classifier.fit(train_features, train_labels)
        predictions = best_rf_classifier.predict(test_features)
        
        # predict_proba çıktısını NumPy dizisine dönüştür
        probabilities = np.array(best_rf_classifier.predict_proba(test_features))[:, 1]
        
        # test_labels ve predictions'ı tek boyutlu hale getir
        test_labels = test_labels.ravel()
        predictions = predictions.ravel()
        
        accuracy = np.mean(predictions == test_labels)
        accuracies.append(accuracy)
        confusion_matrices.append(confusion_matrix(test_labels, predictions))
        
        # Tahminleri doğru bir şekilde birleştir
        all_predictions.extend(predictions.tolist())
        all_probabilities.extend(probabilities.tolist())
        print(f"Fold accuracy: {accuracy}")

    # Uzunlukları kontrol et
    print(f"Length of all_predictions: {len(all_predictions)}")
    print(f"Length of labels_classification: {len(labels_classification)}")

    # Classification Report
    classification_report_str = classification_report(labels_classification[:len(all_predictions)], all_predictions)
    print("Classification Report:")
    print(classification_report_str)

    end_time = time.time()
    mean_accuracy = np.mean(accuracies)
    total_time = end_time - start_time

    print(f"Mean accuracy: {mean_accuracy}")
    print(f"Total run time: {total_time} seconds")

    # Genel karışıklık matrisi
    overall_confusion_matrix = np.sum(confusion_matrices, axis=0)
    print("Overall Confusion Matrix:")
    print(overall_confusion_matrix)

    # ROC eğrisi ve AUC
    fpr, tpr, thresholds = roc_curve(labels_classification[:len(all_probabilities)], all_probabilities, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, roc_auc)

    # En iyi eşik değeri
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    best_threshold = thresholds[ix]
    print(f'Best Threshold={best_threshold}, G-Mean={gmeans[ix]}')

    # Sonuçları bir metin dosyasına yazma
    content = f"""
Istanbul Stock Exchange Veri Seti İlk Satırları:
{data_classification.head()}

Performans Analizi:

Istanbul Stock Exchange Sınıflandırma Problemi

Veri Seti: Istanbul Stock Exchange Veri Seti

Sonuçlar:
- Ortalama doğruluk: {mean_accuracy}
- Toplam çalışma süresi: {total_time} saniye

Genel Karışıklık Matrisi:
{overall_confusion_matrix}

Sınıflandırma Raporu:
{classification_report_str}

Performans Analizi:
- Ortalama doğruluk ({mean_accuracy}), k-katlamalı çapraz doğrulama boyunca elde edilen ortalama doğruluğu ifade eder. Bu değer, modelin veri seti üzerindeki performansını gösterir.
- Toplam çalışma süresi ({total_time} saniye), modelin eğitilmesi ve değerlendirilmesi için geçen süreyi ifade eder. Bu değer, modelin hesaplama açısından verimli olduğunu gösterir.

Algoritma Analizi:
- Random Forest sınıflandırıcısı, {mean_accuracy} ortalama doğruluk ile veri setine iyi bir şekilde uyum sağlamıştır.
- Genel karışıklık matrisi, doğru pozitif, doğru negatif, yanlış pozitif ve yanlış negatif tahminlerin sayısını gösterir. Bu matris, sınıflandırıcının performansını anlamada yardımcı olur.
- Sınıflandırma raporu, her sınıf için kesinlik, duyarlılık ve f1-skoru gibi ayrıntılı metrikler sağlar. Bu metrikler, her sınıfın performansını ayrı ayrı anlamada yardımcı olur.
- ROC eğrisi ve AUC, sınıflandırıcının farklı eşik değerleri üzerindeki performansını görsel olarak temsil eder. En iyi eşik değeri, duyarlılık ve özgüllük arasındaki dengeyi optimize etmek için belirlenmiştir.

Yorumlar ve Notlar:
- Bu analizde kullanılan veri seti, Istanbul Stock Exchange veri setidir ve borsa verileriyle ilgili çeşitli özellikler içermektedir.
- Random Forest sınıflandırıcısı, ortalama doğruluk açısından iyi bir performans göstermiştir ve veri setindeki temel desenleri yakalayabilmiştir.
- ROC eğrisi ve AUC, sınıflandırıcının performansını kapsamlı bir şekilde değerlendirmekte ve en iyi eşik değeri, sınıflandırıcının karar verme sürecini optimize etmektedir.
- Genel olarak, Random Forest sınıflandırıcısı, doğruluk, hesaplama verimliliği ve eşik optimizasyonu açısından bu veri seti için önerilmektedir.
"""

    file_path = "/Users/mervedonmez/Documents/VSProjects/SealExample3/performance_analysis.txt"
    with open(file_path, "w") as file:
        file.write(content)
    print(f"Performance analysis written to {file_path}")

if __name__ == "__main__":
    classification_problem()
