

from math import exp


def relu(lst):

	sonuc = []

	for item in lst:

		temp = max(0,item)

		sonuc.append(temp)

	return sonuc

# THE-3 PDF inde sigmoid clipped için hata var x < -700 olmalı



def sigmoid(lst):

	sonuc = []

	for item in lst:

		if item <= -700:

			temp = 0

			sonuc.append(temp)

		elif -700 < item < 700:

			temp = 1/(1 + exp(-item))

			sonuc.append(temp)						

		elif 700 <= item:

			temp = 1

			sonuc.append(temp)

	return sonuc



def forward_pass(network, vector):

	for layer in network: 

		if type(layer) == list and "linear" in layer[0]:

			weights = layer[1]

			linear_sonucu = []

			for alt_agırlık in weights:

				toplam_tutacak_liste = []

				for i in range(len(vector)):

					temporary_sayi = alt_agırlık[i] * vector[i]

					toplam_tutacak_liste.append(temporary_sayi)

				linear_sonucu.append(sum(toplam_tutacak_liste))	

			vector = linear_sonucu	

		elif "relu" in layer:

			vector = relu(vector)		# Liste olarak vermeyi unutma!

		elif "sigmoid" in layer:

			vector = sigmoid(vector)

	return vector	
