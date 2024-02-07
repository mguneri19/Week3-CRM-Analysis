import datetime as dt
import pandas as pd
pd.set_option('display.max_columns',None)
pd.set_option('display.float_format', lambda x: '%.2f' %x) # virgülden sonra 2 basamaklı olsun

#Görev 1:  Veriyi Anlama ve Hazırlama

#Adım1:   flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz
df_ = pd.read_csv(r"C:\Users\muhammet.guneri\Desktop\FLO-Müşteri Segmentasyonu\flo_data_20k.csv")
df= df_.copy()


#Adım2: Veri setinde;
# a. İlk 10 gözlem,
df.head(10)
# b. Değişken isimleri,
df.columns
# c. Betimsel istatistik,
df.describe().T
# d. Boş değer,
df.isnull().sum() #hiç boş değerim yok.
# e. Değişken tipleri,
df.dtypes #date'ler object burada, bunları datetime'a çevirmem lazım.
df.info()

#Adım3:Omnichannel müşterilerin hem online'dan hem de offline platformlardan
# alışveriş yaptığını ifade etmektedir.
# Her bir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

df["total_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
df.head()

#Adım4:  Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin
# tipini date'e çeviriniz

df.dtypes #4 tane tarih değişkeni var ama veri tipleri object, bunları değiştirmeliyiz
date_columns = [col for col in df.columns if "date" in col] #date içerenleri bul
for col in date_columns:
    df[col] = pd.to_datetime(df[col])  #date içerenleri datetime veri tipine çevir

df.dtypes #kontrolünü yapıyoruz


#Adım5:  Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısının
# ve toplam harcamaların dağılımına bakınız
df.head()

df.groupby("order_channel").agg({"master_id":"nunique",
                                 "order_num_total": "sum",
                                 "total_value": "sum"})
#en fazla müşteri Android App'ten geliyor.


#Adım6:  En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.
df.groupby("master_id").agg({"total_value": "sum"}).sort_values(by="total_value",ascending=False).head(10)


#Adım7:  En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
df.groupby("master_id").agg({"order_num_total": "sum"}).sort_values(by="order_num_total",ascending=False).head(10)

#Adım8:  Veri önhazırlık sürecini fonksiyonlaştırınız.
def data_prep(dataframe):
    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["total_value"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    date_columns = [col for col in dataframe.columns if "date" in col]
    for col in date_columns:
        dataframe[col] = pd.to_datetime(dataframe[col])

    dataframe.groupby("order_channel").agg({"master_id":"nunique",
                                 "order_num_total": "sum",
                                 "total_value": "sum"})

#######################################################
#Görev 2:  RFM Metriklerinin Hesaplanması

#Adım 1: Recency, Frequency ve Monetary tanımlarını yapınız.
#Adım 2: Müşteri özelinde Recency, Frequency ve Monetary metriklerini hesaplayınız.
#Adım 3: Hesapladığınız metrikleri rfm isimli bir değişkene atayınız

df.describe().T #last order date = 2021-05-30, 2gün ekliyorum analiz zamanına
today_date = dt.datetime(2021,6,1)
rfm = df.groupby("master_id").agg({"last_order_date": lambda date: (today_date-date.max()).days,
                                   "order_num_total": lambda num: num.sum(),
                                   "total_value": lambda x: x.sum()})
rfm.reset_index()
rfm.columns = ["recency", "frequency", "monetary"]
rfm.describe().T

##############################################################

#Görev 3:  RF Skorunun Hesaplanması

#Adım 1: Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çeviriniz
#Adım 2: Bu skorları recency_score, frequency_scoreve monetary_score olarak kaydediniz.

rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1]) # recency'deki kısmı 5'e böl, küçük rakamlılara 5 ver, büyüklere 1 ver.
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])
rfm.head()

#Adım 3: recency_score ve frequency_score’utek bir değişken olarak ifade ediniz ve RF_SCORE olarak kaydediniz.
rfm["RF_SCORE"] = (rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str))
rfm.head()

#######################################################################

#Görev 4:  RF Skorunun Segment Olarak Tanımlanması

#RF isimlendirilmesi
#Adım 1: Oluşturulan RF skorları için segment tanımlamaları yapınız.
seg_map = {
          r'[1-2][1-2]': "hibernating",
          r'[1-2][3-4]': "at_Risk",
          r'[1-2]5': "cant_loose",
          r'3[1-2]': "about_to_sleep",
          r'33': "need_attention",
          r'[3-4][4-5]': "loyal_customers",
          r'41': "promising",
          r'[4-5][2-3]': "potential_loyalists",
          r'5[4-5]': "champions",
          r'51': "new_customers"

}


#Adım 2: Aşağıdaki seg_map yardımı ile skorları segmentlere çeviriniz.

rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)
rfm.head()


#########################################################################

#Görev 5:  Aksiyon Zamanı

#Adım1:  Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg("mean")

#Adım2:  RFM analizi yardımıyla aşağıda verilen 2 case için ilgili profildeki müşterileri
# bulun ve müşteri id'lerini csv olarak kaydediniz

#a.FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor.
# Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerin in üstünde.
# Bu nedenle markanın tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle
# özel olarak iletişime geçmek isteniliyor.Sadık müşterilerinden(champions,loyal_customers)
# ve kadın kategorisinden alışveriş yapan kişiler özel olarak iletişim kurulacak müşteriler.
# Bu müşterilerin id numaralarını csv dosyasına kaydediniz.
df.head()

merged_df = pd.merge(df, rfm, on="master_id") #iki ayrı veri tabanındaki veriyi birleştirdim.
targeted_customers = merged_df[(merged_df["segment"].isin(["champions", "loyal_customers"]) &
                      merged_df["interested_in_categories_12"].str.contains("KADIN"))]
targeted_customers["master_id"].reset_index()
print(targeted_customers["master_id"].reset_index())

targeted_customers["master_id"].to_csv("targeted_customers.csv")

# b.Erkek ve Çocukürünlerinde %40'a yakın indirim planlanmaktadır.
# Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşteri olan ama uzun süredir alışveriş
# yapmayan kaybedilmemesi gereken müşteriler,uykuda olanlar ve yeni gelen müşteriler özel olarak hedef alınmak isteniyor.
# Uygun profildeki müşterilerin id'lerini csv dosyasına kaydediniz.
rfm.head()
df.head()
rfm["segment"].value_counts()
target_customers2 = merged_df[
    (merged_df["segment"].isin(["new_customers", "about_to_sleep"])) &
    (merged_df["interested_in_categories_12"].str.contains("ERKEK|COCUK", case=False))
]

target_customers2
print(target_customers2["master_id"].reset_index()) #947 müşteri

target_customers2["master_id"].to_csv("target_customers2.csv")