import seal
import numpy as np

# Şifreleme parametrelerini ayarla
parms = seal.EncryptionParameters(seal.scheme_type.ckks)
poly_modulus_degree = 8192
parms.set_poly_modulus_degree(poly_modulus_degree)
parms.set_coeff_modulus(seal.CoeffModulus.Create(poly_modulus_degree, [60, 40, 40, 60]))

# SEALContext oluştur
context = seal.SEALContext(parms)

# Anahtar üretici oluştur
keygen = seal.KeyGenerator(context)

# PublicKey ve SecretKey oluştur
public_key = keygen.create_public_key()  # PublicKey oluştur
secret_key = keygen.secret_key()  # SecretKey doğrudan alınır

# Şifreleyici ve çözücü oluştur
encryptor = seal.Encryptor(context, public_key)
decryptor = seal.Decryptor(context, secret_key)

# Değerlendirici oluştur
evaluator = seal.Evaluator(context)

# CKKS kodlayıcı oluştur
encoder = seal.CKKSEncoder(context)

# Ölçek ayarla
scale = 2**40

# Düz metin oluştur
value = np.array([3.14159265])  # value'yu bir numpy dizisine dönüştür
plain_text = encoder.encode(value, scale)  # encode fonksiyonu bir Plaintext döndürür

# Düz metni şifrele
cipher_text = encryptor.encrypt(plain_text)  # encrypt fonksiyonu bir Ciphertext döndürür

# Şifreli metni çöz
plain_result = seal.Plaintext()
decryptor.decrypt(cipher_text, plain_result)

# Çözülmüş düz metni decode et
decoded_result = encoder.decode(plain_result)

# Sonucu yazdır
print("Decrypted result:", decoded_result[0])