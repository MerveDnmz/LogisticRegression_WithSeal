def count_lines_in_veriler_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Şifrelenmemiş ve şifrelenmiş veriler için başlangıç ve bitiş indekslerini bul
    unencrypted_start = lines.index("Şifrelenmemiş Veriler:\n") + 1
    encrypted_start = lines.index("Şifrelenmiş Veriler:\n") + 1  # Fazladan boşluk kaldırıldı

    # Şifrelenmemiş ve şifrelenmiş veri satırlarını ayır
    unencrypted_lines = lines[unencrypted_start:encrypted_start - 1]
    encrypted_lines = lines[encrypted_start:]

    # Satır sayılarını hesapla
    unencrypted_count = len(unencrypted_lines)
    encrypted_count = len(encrypted_lines)

    return unencrypted_count, encrypted_count

# Dosya yolu
file_path = "/Users/mervedonmez/Documents/VSProjects/SealExample3/LogisticRegression_WithSeal_veriler.txt"

# Satır sayılarını hesapla
unencrypted_count, encrypted_count = count_lines_in_veriler_file(file_path)

# Sonuçları yazdır
print(f"Şifrelenmemiş veri satır sayısı: {unencrypted_count}")
print(f"Şifrelenmiş veri satır sayısı: {encrypted_count}")