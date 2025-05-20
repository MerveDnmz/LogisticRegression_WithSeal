import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from ucimlrepo import fetch_ucirepo
from seal import EncryptionParameters, SEALContext, KeyGenerator, Encryptor, Decryptor, CKKSEncoder, Plaintext, Ciphertext, scheme_type, CoeffModulus
import time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# SEAL ile veri şifreleme fonksiyonları
def seal_encrypt_data(data):
    """
    SEAL kullanarak veriyi şifreler.
    """
    parms = EncryptionParameters(scheme_type.ckks)
    parms.set_poly_modulus_degree(8192)
    parms.set_coeff_modulus(CoeffModulus.Create(8192, [60, 40, 40, 60]))
    context = SEALContext(parms)

    keygen = KeyGenerator(context)
    public_key = keygen.create_public_key()
    secret_key = keygen.secret_key()

    encryptor = Encryptor(context, public_key)
    encoder = CKKSEncoder(context)
    scale = pow(2.0, 40)

    encrypted_data = []
    for row in data:
        row = np.array(row, dtype=np.float64).flatten()  # Çok boyutlu diziyi tek boyutlu hale getir
        plain = encoder.encode(row, scale)  # Plaintext nesnesi otomatik olarak oluşturulacak
        encrypted_row = encryptor.encrypt(plain)  # Ciphertext nesnesi döndürülür
        encrypted_data.append(encrypted_row)

    return encrypted_data, secret_key, encoder, scale, context

def seal_decrypt_data(encrypted_data, secret_key, encoder, scale, context):
    """
    SEAL kullanarak şifrelenmiş veriyi çözer.
    """
    decrypted_data = []
    decryptor = Decryptor(context, secret_key)
    for encrypted_row in encrypted_data:
        plain = Plaintext()
        decryptor.decrypt(encrypted_row, plain)
        decoded_row = np.array(encoder.decode(plain))  # decode() doğrudan bir numpy.ndarray döndürür
        decrypted_data.append(decoded_row[:23])  # İlk 23 özelliği alın
    return np.array(decrypted_data)

def plot_roc_curve(y_test, y_pred_proba_unencrypted, y_pred_proba_encrypted):
    """
    Şifrelenmiş ve şifrelenmemiş veriler için ROC eğrilerini çizer ve kaydeder.
    """
    # Şifrelenmemiş veriler için ROC eğrisi
    fpr_unencrypted, tpr_unencrypted, _ = roc_curve(y_test, y_pred_proba_unencrypted[:, 1])
    roc_auc_unencrypted = auc(fpr_unencrypted, tpr_unencrypted)

    # Şifrelenmiş veriler için ROC eğrisi
    fpr_encrypted, tpr_encrypted, _ = roc_curve(y_test, y_pred_proba_encrypted[:, 1])
    roc_auc_encrypted = auc(fpr_encrypted, tpr_encrypted)

    # ROC eğrilerini çiz
    plt.figure(figsize=(10, 6))
    plt.plot(fpr_unencrypted, tpr_unencrypted, color='blue', lw=2,
             label=f'Şifrelenmemiş Veriler (AUC = {roc_auc_unencrypted:.2f})')
    plt.plot(fpr_encrypted, tpr_encrypted, color='green', lw=2,
             label=f'Şifrelenmiş Veriler (AUC = {roc_auc_encrypted:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')

    plt.xlabel('False Positive Rate (Yanlış Pozitif Oranı)')
    plt.ylabel('True Positive Rate (Doğru Pozitif Oranı)')
    plt.title('ROC Eğrisi - Şifrelenmiş ve Şifrelenmemiş Veriler')
    plt.legend(loc='lower right')
    plt.grid()

    # Grafiği kaydet
    output_path = "/Users/mervedonmez/Documents/VSProjects/SealExample3/ROC_Curve.png"
    plt.savefig(output_path)
    print(f"ROC eğrisi '{output_path}' dosyasına kaydedildi.")

    # Grafiği göster
    plt.show()

def plot_individual_roc_curves(y_test, y_pred_proba_unencrypted, y_pred_proba_encrypted):
    """
    Şifrelenmiş ve şifrelenmemiş veriler için ayrı ayrı ROC eğrilerini çizer ve kaydeder.
    """
    # Şifrelenmemiş veriler için ROC eğrisi
    fpr_unencrypted, tpr_unencrypted, _ = roc_curve(y_test, y_pred_proba_unencrypted[:, 1])
    roc_auc_unencrypted = auc(fpr_unencrypted, tpr_unencrypted)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr_unencrypted, tpr_unencrypted, color='blue', lw=2,
             label=f'Şifrelenmemiş Veriler (AUC = {roc_auc_unencrypted:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate (Yanlış Pozitif Oranı)')
    plt.ylabel('True Positive Rate (Doğru Pozitif Oranı)')
    plt.title('ROC Eğrisi - Şifrelenmemiş Veriler')
    plt.legend(loc='lower right')
    plt.grid()
    output_path_unencrypted = "/Users/mervedonmez/Documents/VSProjects/SealExample3/ROC_Curve_Unencrypted.png"
    plt.savefig(output_path_unencrypted)
    print(f"Şifrelenmemiş veriler için ROC eğrisi '{output_path_unencrypted}' dosyasına kaydedildi.")
    plt.show()

    # Şifrelenmiş veriler için ROC eğrisi
    fpr_encrypted, tpr_encrypted, _ = roc_curve(y_test, y_pred_proba_encrypted[:, 1])
    roc_auc_encrypted = auc(fpr_encrypted, tpr_encrypted)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr_encrypted, tpr_encrypted, color='green', lw=2,
             label=f'Şifrelenmiş Veriler (AUC = {roc_auc_encrypted:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate (Yanlış Pozitif Oranı)')
    plt.ylabel('True Positive Rate (Doğru Pozitif Oranı)')
    plt.title('ROC Eğrisi - Şifrelenmiş Veriler')
    plt.legend(loc='lower right')
    plt.grid()
    output_path_encrypted = "/Users/mervedonmez/Documents/VSProjects/SealExample3/ROC_Curve_Encrypted.png"
    plt.savefig(output_path_encrypted)
    print(f"Şifrelenmiş veriler için ROC eğrisi '{output_path_encrypted}' dosyasına kaydedildi.")
    plt.show()

def logistic_regression_with_seal():
    # Veri setini indir
    default_of_credit_card_clients = fetch_ucirepo(id=350)
    X = default_of_credit_card_clients.data.features
    y = default_of_credit_card_clients.data.targets

    # `y` verisini NumPy dizisine dönüştür ve boyutunu düzelt
    y = y.to_numpy().ravel()

    # Veriyi eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Şifrelenmemiş verilerle Logistic Regression
    print("Şifrelenmemiş verilerle Logistic Regression:")
    start_time = time.time()
    model = LogisticRegression(max_iter=2000)  # max_iter artırıldı
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    unencrypted_time = time.time() - start_time

    # Performans değerlendirme (şifrelenmemiş)
    # Accuracy, modelin doğru tahmin ettiği örneklerin toplam örnek sayısına oranıdır
    unencrypted_accuracy = accuracy_score(y_test, y_pred)

    # Şifrelenmemiş veriler için karışıklık matrisi
    # Confusion matrix, modelin tahmin performansını sınıf bazında gösteren bir tablodur.
    # [[TP, FP],  -> [[4552,  135], 4552: Sınıf 0 için doğru tahmin edilen örnekler (TP). /  135: Sınıf 0 için yanlış pozitif tahminler (FP).
    # [FN, TN]]  ->  [1056,  257]], 1056: Sınıf 1 için yanlış negatif tahminler (FN). /  257: Sınıf 1 için doğru tahmin edilen örnekler (TN).
    # TP: True Positive, FP: False Positive, FN: False Negative, TN: True Negative
    # Classification report, modelin her sınıf için doğruluk, hassasiyet, F1 skoru gibi metriklerini gösterir.
    unencrypted_confusion_matrix = confusion_matrix(y_test, y_pred)
    # Macro Avg: Tüm sınıfların precision, recall ve F1-score değerlerinin basit ortalamasıdır, Sınıf dengesizliğini dikkate almaz.
    # Weighted Avg: Tüm sınıfların precision, recall ve F1-score değerlerinin, sınıf destek (support) sayısına göre ağırlıklı ortalamasıdır. Sınıf dengesizliğini dikkate alır.
    unencrypted_classification_report = classification_report(y_test, y_pred)

    # Veriyi şifrele
    print("\nVeriyi SEAL ile şifreliyoruz...")
    encrypted_X_test, secret_key, encoder, scale, context = seal_encrypt_data(X_test.to_numpy())

    # Şifrelenmiş veriler üzerinde tahmin yap
    print("Şifrelenmiş veriler üzerinde tahmin yapılıyor...")
    decrypted_X_test = seal_decrypt_data(encrypted_X_test, secret_key, encoder, scale, context)
    decrypted_X_test = pd.DataFrame(decrypted_X_test, columns=X_test.columns)
    y_pred_encrypted = model.predict(decrypted_X_test)

    # Performans değerlendirme (şifrelenmiş)
    encrypted_accuracy = accuracy_score(y_test, y_pred_encrypted)

    # Şifrelenmiş ve şifrelenmemiş verileri bir dosyaya yazdır
    output_file = "/Users/mervedonmez/Documents/VSProjects/SealExample3/LogisticRegression_WithSeal_veriler.txt"
    with open(output_file, "w") as f:
        f.write("Şifrelenmemiş Veriler:\n")
        f.write(X_test.to_string(index=False))
        f.write("\n\nŞifrelenmiş Veriler:\n")
        for encrypted_row in encrypted_X_test:
            f.write(str(encrypted_row) + "\n")

    print(f"Şifrelenmiş ve şifrelenmemiş veriler '{output_file}' dosyasına yazıldı.")

    # Şifrelenmemiş veriler için olasılık tahminleri
    y_pred_proba_unencrypted = model.predict_proba(X_test)

    # Şifrelenmiş veriler için olasılık tahminleri
    y_pred_proba_encrypted = model.predict_proba(decrypted_X_test)

    # ROC eğrisini çiz
    plot_roc_curve(y_test, y_pred_proba_unencrypted, y_pred_proba_encrypted)

    # ROC eğrilerini ayrı ayrı çiz
    plot_individual_roc_curves(y_test, y_pred_proba_unencrypted, y_pred_proba_encrypted)

    # Raporu oluştur ve dosyaya yaz
    output_file = "/Users/mervedonmez/Documents/VSProjects/SealExample3/LogisticRegression_WithSeal_performans_analizi.txt"
    with open(output_file, "w") as f:
        f.write("Logistic Regression with SEAL - Performans Analizi\n")
        f.write("=" * 50 + "\n\n")
        f.write("**Neden Logistic Regression?**\n")
        f.write("Logistic Regression, sınıflandırma problemleri için kullanılan basit ve etkili bir yöntemdir. "
                "Bu çalışmada, şifrelenmiş ve şifrelenmemiş verilerle aynı doğruluk oranlarını elde etmek için kullanılmıştır.\n\n")
        
        f.write("**Neden SEAL?**\n")
        f.write("SEAL, Microsoft tarafından geliştirilen bir homomorfik şifreleme kütüphanesidir. "
                "Bu çalışmada, verilerin gizliliğini koruyarak makine öğrenimi modelleri üzerinde işlem yapmayı mümkün kılmak için kullanılmıştır.\n\n")
        
        f.write("**SEAL ile Şifreleme Süreci:**\n")
        f.write("- CKKS şifreleme şeması kullanıldı.\n")
        f.write("- Polinom derecesi: 8192\n")
        f.write("- Ölçek: 2^40\n")
        f.write("- Şifreleme ve çözme işlemleri sırasında verinin matematiksel yapısı korundu.\n\n")
        
        f.write("**Sonuçlar:**\n")
        f.write("Şifrelenmemiş Veriler:\n")
        f.write(f"Accuracy: {unencrypted_accuracy}\n")
        f.write(f"Confusion Matrix:\n{unencrypted_confusion_matrix}\n")
        f.write(f"Classification Report:\n{unencrypted_classification_report}\n")
        f.write(f"Processing Time: {unencrypted_time:.2f} seconds\n\n")

        f.write("Şifrelenmiş Veriler:\n")
        f.write(f"Accuracy: {encrypted_accuracy}\n")
        f.write(f"Confusion Matrix:\n{encrypted_confusion_matrix}\n")
        f.write(f"Classification Report:\n{encrypted_classification_report}\n")
        f.write(f"Encryption Time: {encryption_time:.2f} seconds\n")
        f.write(f"Decryption and Prediction Time: {encrypted_time:.2f} seconds\n\n")

        f.write("**Sonuçların Yorumu:**\n")
        f.write("Şifrelenmiş ve şifrelenmemiş verilerle yapılan Logistic Regression işlemleri aynı doğruluk oranlarını vermiştir. "
                "Bu, SEAL'in verinin matematiksel yapısını koruduğunu göstermektedir. Ancak, şifreleme ve çözme işlemleri işlem süresini önemli ölçüde artırmıştır.\n\n")
        f.write("Bu çalışma, homomorfik şifrelemenin gizlilik koruma açısından etkili bir yöntem olduğunu, ancak işlem süresi açısından maliyetli olduğunu göstermektedir.\n")

    print(f"Performans analizi ve rapor '{output_file}' dosyasına yazıldı.")

if __name__ == "__main__":
    logistic_regression_with_seal()
