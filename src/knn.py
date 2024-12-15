from statistics import mode
import pickle
import numpy as np

class KNearestNeighbors:
    def __init__(self, jumlah_neighbor:int = 5, r:int = 2) -> None:
        """
        r = 1 => manhattan
        r = 2 => euclidian
        r >= 3 => minkowski
        """
        self.jumlah_neighbor = jumlah_neighbor
        self.r = r
        self.data_x = []
        self.data_y = []

    def hitung_jarak(self, titik1:list[int], titik2:list[int]) -> int | Exception:
        if len(titik2) != len(titik1):
            return Exception("Tidak dapat menghitung jarak antara 2 titik yang berbeda dimensi")
        jaraknya = 0
        for i in range(len(titik1)):
            jaraknya += (abs(titik2[i] - titik1[i]))**(self.r)
        return (jaraknya**(1/self.r))
 
    def fit(self, x_train:list[list[int]], y_train:list[list[int]]) -> None:
        self.data_x.extend(x_train)
        self.data_y.extend(y_train)

    def predict_point(self, x_pred:list[int]) -> list[int]:
        list_jarak = []
        list_hasil = []
        for i in range(len(self.data_x)):
            jaraknya = self.hitung_jarak(x_pred, self.data_x[i])
            dimasukin = False
            i = 0
            while(not dimasukin and i < len(list_jarak)):
                if list_jarak[i] > jaraknya:
                    list_jarak.insert(i, jaraknya)
                    list_hasil.insert(i, tuple(self.data_y[i]))
                    dimasukin = True
                i+=1
            if not dimasukin:
                list_jarak.append(jaraknya)
                list_hasil.append(tuple(self.data_y[i]))
        toberet = mode(list_hasil[:(self.jumlah_neighbor)])
        return np.array(toberet)

    def predict(self, x_pred_list:list[list[int]]) -> list[list[int]]:
        hasilnya = []
        for titik in x_pred_list:
            hasilnya.append(self.predict_point(titik))
        return np.array(hasilnya)

    def save(self, filename) -> None:
        try:
            with open(filename, "wb") as filenya:
                pickle.dump(self, filenya)
        except Exception as err:
            print(err)
    
    @classmethod
    def load(cls, filename):
        try:
            with open(filename, "rb") as filenya:
                return (pickle.load(filenya))
        except Exception as err:
            print(err)