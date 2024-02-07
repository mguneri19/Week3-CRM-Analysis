import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
pd.set_option('display.max_columns',None)
pd.set_option('display.width',500)
pd.set_option('display.float_format', lambda x: '%.2f' %x) # virgülden sonra 2 basamaklı olsun
from sklearn.preprocessing import MinMaxScaler

#Görev 1:  Veriyi Hazırlama

#Adım1:   flo_data_20K.csv verisini okuyunuz.
df_ = pd.read_csv(r"C:\Users\muhammet.guneri\Desktop\FLO-Müşteri Segmentasyonu\flo_data_20k.csv")
df = df_.copy()
df.head()

#Adım2:  Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını
# tanımlayınız. Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.
# Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.

def outlier_thresholds(dataframe,variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds (dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    rounded_values = dataframe[variable].apply(lambda x: round(x)) #round ile yuvarlama işlemi
    dataframe.loc[rounded_values < low_limit, variable] = low_limit
    dataframe.loc[rounded_values > up_limit, variable] = up_limit


#Adım3:  "order_num_total_ever_online", "order_num_total_ever_offline",
# "customer_value_total_ever_offline", "customer_value_total_ever_online"
# değişkenlerini naykırı değerleri varsa baskılayanız.
df.head()
df.describe().T

replace_with_thresholds(df,"order_num_total_ever_online")
replace_with_thresholds(df,"order_num_total_ever_offline")
replace_with_thresholds(df,"customer_value_total_ever_offline")
replace_with_thresholds(df,"customer_value_total_ever_online")
df.describe().T

#Adım4:  Omnichannel müşterilerin hem online'dan hem de offline platformlardan
# alışveriş yaptığını ifade etmektedir. Her bir müşterinin toplam alışveriş sayısı ve
# harcaması için yeni değişkenler oluşturunuz.

df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_value"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
df.head()


#Adım5:  Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz
df.dtypes #date'leri datetime yapmalıyım.
date_columns = [col for col in df.columns if "date" in col]
for col in date_columns:
    df[col] = pd.to_datetime(df[col])

df.dtypes

####################################################################
#Görev 2:  CLTV Veri Yapısının Oluşturulması

#Adım1:  Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.

df.describe().T
today_date = dt.datetime(2021,6,1) #analiz tarihi alışveriş tarihin en son tarihine 2 eklendi.

# Adım2:  customer_id, recency_cltv_weekly, T_weekly, frequency vemonetary_cltv_avg değerlerinin yer aldığı yeni bir cltv data frame'i oluşturunuz.
# Monetary değeri satınalma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten
# ifade edilecek.
df.head()


df['recency'] = (df['last_order_date'] - df['first_order_date']).dt.days

df['frequency'] = df['order_num_total']

df['T'] = (today_date - df['first_order_date']).dt.days #müşteri yaşı

df['monetary_avg'] = df['total_value'] / df['frequency']

df['recency_cltv_weekly'] = df['recency'] / 7
df['T_weekly'] = df['T'] / 7

cltv_df = df[['master_id', 'recency_cltv_weekly', 'T_weekly', 'frequency', 'monetary_avg']]

cltv_df.head()

###############################################################################
#Görev 3:  BG/NBD, Gamma-Gamma Modellerinin Kurulması ve CLTV’nin Hesaplanması

#Adım1:  BG/NBD modelini fit ediniz.

#3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve
# exp_sales_3_month olarak cltv data frame'ine ekleyiniz

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df["frequency"],
        cltv_df["recency_cltv_weekly"],
        cltv_df["T_weekly"])


cltv_df["exp_sales_3_month"] = bgf.predict(12,
            cltv_df["frequency"],
            cltv_df["recency_cltv_weekly"],
            cltv_df["T_weekly"]
            ).sort_values(ascending=False)


#6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve
# exp_sales_6_month olarak cltv dataframe'ine ekleyiniz

cltv_df["exp_sales_6_month"] = bgf.predict(24,
            cltv_df["frequency"],
            cltv_df["recency_cltv_weekly"],
            cltv_df["T_weekly"]
            ).sort_values(ascending=False)


#Adım2:  Gamma-Gamma modelinifit ediniz. Müşterilerin ortalama bırakacakları değeri
# tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz

ggf = GammaGammaFitter(penalizer_coef=0.01)
cltv_df["frequency"] = cltv_df["frequency"].round().astype(int) # hata verdi bu nedenle non-integerleri integer'e çeviriyoruz
ggf.fit(cltv_df["frequency"], cltv_df["monetary_avg"])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                       cltv_df["monetary_avg"])

#Adım3:  6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.

cltv_df["CLTV"] = ggf.customer_lifetime_value(bgf,
                                              cltv_df["frequency"],
                                              cltv_df["recency_cltv_weekly"],
                                              cltv_df["T_weekly"],
                                              cltv_df["monetary_avg"],
                                              time=6, # 6 aylık
                                              freq="W", # T'nin frekans bilgisi
                                              discount_rate=0.01)

#Cltv değeri en yüksek 20 kişiyi gözlemleyiniz
cltv_df.sort_values(by="CLTV", ascending=False).head(20)


###############################################################################################

#Görev 4:  CLTV Değerine Göre Segmentlerin Oluşturulması

#Adım1:  6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba(segmente) ayırınız ve grup isimlerini veri setine ekleyiniz

cltv_df["segment"] = pd.qcut(cltv_df["CLTV"],4, labels=["D", "C", "B", "A"])


#Adım2:  4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz.

cltv_df.groupby("segment").agg({"CLTV":["count", "mean", "sum"]})

#Segment A: En yüksek ortalama CLTV'ye (362.32) sahip bu grup, en değerli müşterileri temsil ediyor.
#Segment C: Orta düzeyde bir ortalama CLTV'ye (138.31) sahip olan bu grup, gelişim potansiyeli taşıyan bir segment.
#Segment A İçin Aksiyon Önerileri (6 Aylık):

#Sadakati Artırma Programları: Bu müşterilere yönelik özel sadakat programları ve ödüller tasarlayın.
   # Örneğin, özel indirimler, üyelik avantajları veya özel etkinlik davetleri sunarak sadakatlerini pekiştirin.
   # Pazarlama: Müşterilerin alışveriş tercihlerine ve geçmiş davranışlarına dayalı kişiselleştirilmiş ürün önerileri
   # ve iletişim stratejileri geliştirin.
   #Premium Hizmetler: Bu segment için premium hizmetler veya ürünler sunarak, onların deneyimini zenginleştirin.
   # Örneğin, özel müşteri hizmetleri desteği veya üst düzey ürünler sunabilirsiniz.


#Segment C İçin Aksiyon Önerileri (6 Aylık):

#Müşteri Katılımını Artırma: E-posta kampanyaları, sosyal medya etkileşimleri ve kişiselleştirilmiş içeriklerle
# bu grubun katılımını artırın. Bu, marka bilincini artıracak ve satın alma sıklığını teşvik edebilir.
#Ürün ve Hizmet Çeşitliliğini Artırma: Bu segmentteki müşterilere çeşitli ürün ve hizmetler sunarak, ihtiyaçlarına daha
# iyi cevap verin. Bu, onların daha fazla harcama yapmalarını teşvik edebilir.
#Geri Bildirim ve Anketler: Müşteri geri bildirimlerini toplayarak ve anketler düzenleyerek, bu grubun tercihlerini
# ve ihtiyaçlarını daha iyi anlayın. Bu bilgiler, ürün ve hizmetlerinizi bu segmente daha iyi uyarlamanıza yardımcı olur.