# uav-threat-avoidance-rl

Hareketli tehditlerden kaçınmayı PPO ile öğrenen ve öğrenilen politikayı kural tabanlı
bir yöntemle aynı senaryolarda karşılaştıran küçük bir öğrenme projesi.

**Soru:** Simülasyonda başarılı görünen bir politika, gerçek bir hava aracının yapamayacağı
manevralara mı dayanıyor?

## Kurulum
2B alanda sabit hızla uçan bir İHA, 5 hareketli tehdit arasından hedefe ulaşmaya çalışır.
Ajan yalnızca dönüş hızını seçer. Uçuş zarfı iki sınırla temsil edilir: en büyük dönüş hızı
(|ω| ≤ 0.15) ve dönüş hızındaki en büyük değişim (|Δω| ≤ 0.05). Zarf açıkken komutlar bu
sınırlara kırpılır, kapalıyken ihlaller sayılır.

| Yöntem | Eğitim | Değerlendirme |
|---|---|---|
| kural_tabanli | potansiyel alanı | zarf açık |
| ppo_zarfli | zarf açık | zarf açık |
| ppo_zarfsiz | zarf kapalı | zarf kapalı |
| ppo_zarfsiz_kalkanli | zarf kapalı | zarf açık |

Tüm yöntemler, eğitimde kullanılmayan aynı 500 senaryoda çalıştırılır. Oranlar Wilson %95
güven aralığıyla verilir, farklar eşleştirilmiş McNemar testiyle sınanır ve yöntemlerin
ayrıştığı senaryoların rotaları çizilir.

## Sonuçlar
<!-- evaluate.py çıktısı ve results/disagreements.png -->

## Çalıştırma
```bash
pip install -r requirements.txt
python train.py --envelope shield
python train.py --envelope none
python evaluate.py
```
