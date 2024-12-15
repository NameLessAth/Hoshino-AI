import numpy as np

class NaiveBayes:
    def __init__(self):
        self.data_x = []
        self.data_y = []
        self.freq_muncul = {}
        self.freq_muncul_hasil = {}
        self.y_unique = []

    def fit(self, x_train:list[list[int]], y_train:list[list[int]]):
        self.data_x.extend(x_train)
        self.data_y.extend([tuple(y_list) for y_list in y_train])
        for i in range(len(self.data_x)):
            j = 1
            for elemen_column in self.data_x[i]:
                column_sekarang = "column" + str(j)
                if (column_sekarang) not in self.freq_muncul:
                    self.freq_muncul[column_sekarang] = {}
                if elemen_column not in self.freq_muncul[column_sekarang]:
                    self.freq_muncul[column_sekarang][elemen_column] = {}
                if self.data_y[i] not in self.freq_muncul[column_sekarang][elemen_column]:
                    self.freq_muncul[column_sekarang][elemen_column][self.data_y[i]] = 0
                self.freq_muncul[column_sekarang][elemen_column][self.data_y[i]] += 1
                j += 1
        for j in self.data_y:
            if j not in self.freq_muncul_hasil:
                self.freq_muncul_hasil[j] = 0
            self.freq_muncul_hasil[j] += 1
            if j not in self.y_unique:
                self.y_unique.append(j)
    
    def predict_point(self, x_pred:list[int]) -> list[int]:
        proba_each = []
        hasil_each = []
        for kemungkinan_hasil in self.y_unique:
            hasil_each.append(kemungkinan_hasil)
            j = 1
            proba_sekarang = 1
            for column in x_pred:
                column_sekarang = "column" + str(j)
                try:
                    proba_sekarang *= (self.freq_muncul[column_sekarang][column][kemungkinan_hasil] / self.freq_muncul_hasil[kemungkinan_hasil])
                except:
                    proba_sekarang *= 0
                    break
                j+=1
            proba_sekarang *= (self.freq_muncul_hasil[kemungkinan_hasil])/len(self.data_y)
            proba_each.append(proba_sekarang)
        proba_terpilih = -1
        hasil_terpilih = self.y_unique[0]
        for i in range(len(hasil_each)):
            if proba_each[i] > proba_terpilih:
                proba_terpilih = proba_each[i]
                hasil_terpilih = hasil_each[i]
        return np.array(hasil_terpilih)

    def predict(self, x_pred_list:list[list[int]]) -> list[list[int]]:
        hasilnya = []
        for titik in x_pred_list:
            hasilnya.append(self.predict_point(titik))
        return np.array(hasilnya)
