# uzycie
# python text_recognition.py --east frozen_east_text_detection.pb --image zdjecia/costam.jpg


# import paczek
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2

def decode_predictions(scores, geometry):
	# wczytaj parametr scores i policz ilosc wierszy oraz kolumn. Potem inicjalizuj ramki opisujace oraz wyniki prawdopodobienstwa
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loopuj po wierszach
	for y in range(0, numRows):
		# wyciągnij prawdopodobieństo, oraz ksztalt, aby go obrysować jeśli jest taka możliwość
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loopuj po kolumnach
		for x in range(0, numCols):
			# jeśli nasz wyniki nie ma wystarczającego prawdopodobieństwa to go zignoruj
			if scoresData[x] < args["min_confidence"]:
				continue

			# oblicz współczynnik przesunięcia, ponieważ nasze wynikowe mapy obiektów będą czterokrotnie mniejsze niż obraz wejściowy
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# wyciągnij kąty obiektów oraz policz ich sin/cos
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# użyj geometrii aby obliczyc jak duza musi byc ramka
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# obliczyć początkowe i końcowe współrzędne (x, y) dla ramki tekstu
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# dodaj współrzędne ramki granicznej i wynik prawdopodobieństwa do listy obiektów
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# zwróc tupla z danymi czyli ramkami i prawdopodobienstwem
	return (rects, confidences)

# parser argumentow
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
	help="sciezka do zdjecia")
ap.add_argument("-east", "--east", type=str,
	help="sciezka do silnika detekcji tekstu EAST")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimalne prawdopodobienstwo sprawdzania regionu - zwykle 0.5")
ap.add_argument("-w", "--width", type=int, default=320,
	help="szerokosc zdjecia - wieloktrotnosc 32")
ap.add_argument("-e", "--height", type=int, default=320,
	help="wysokosc zdjecia - wielokrotnosc 32")
ap.add_argument("-p", "--padding", type=float, default=0.0,
	help="kompensacja ROI w dopelnieniu")
args = vars(ap.parse_args())

# wczytujemy do opencv zdjecia i pobieramy jego wymiary
image = cv2.imread(args["image"])
orig = image.copy()
(origH, origW) = image.shape[:2]

# ustaw nową szerokość i wysokość, a następnie określ zmienny stosunek zarówno dla szerokości, jak i wysokości
(newW, newH) = (args["width"], args["height"])
rW = origW / float(newW)
rH = origH / float(newH)

# zmień rozmiar obrazu i pobierz nowe wymiary obrazu
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# zdefiniuj dwie warswt modelu EAST - detekcji - Sigmoid i ciecia - concat
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# zaladuj silnik detekcji EAST
print("[INFO] wczytuje silnik detekcji tekstu EAST...")
net = cv2.dnn.readNet(args["east"])

# skonstruuj bloba i przekaz do dalej. Dla każdej komutacji wykonaj porównanie rgb
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)

# deserializuj znaliziska i zabezpiecz ramki w pozytywnych predykcjach
(rects, confidences) = decode_predictions(scores, geometry)
boxes = non_max_suppression(np.array(rects), probs=confidences)

# inicjalizuj liste
results = []

# loopuj po ramkach
for (startX, startY, endX, endY) in boxes:
	# skaluj ramke aby opisywala element
	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)

	# w celu uzyskania lepszego OCR tekstu możemy potencjalnie zastosować nieco dopełnienia otaczającego ramkę - tutaj obliczamy delty w obu kierunkach xi y
	dX = int((endX - startX) * args["padding"])
	dY = int((endY - startY) * args["padding"])

	# zastosuje dopelnienie po kazdej ze stron ramki
	startX = max(0, startX - dX)
	startY = max(0, startY - dY)
	endX = min(origW, endX + (dX * 2))
	endY = min(origH, endY + (dY * 2))

	# wyciagnij aktualne dopelnienie ROI
	roi = orig[startY:endY, startX:endX]

	# w celu zastosowania Tesseract v4 do tekstu OCR musimy dostarczyć (1) język, (2) flagę OEM 4, wskazującą, że chcemy użyć modelu sieci neuronowej LSTM dla OCR, a na koniec (3) wartość OEM , w tym przypadku 7, co oznacza, że traktujemy ROI jako pojedynczą linię tekstu
	config = ("-l pol --oem 1 --psm 7")
	text = pytesseract.image_to_string(roi, config=config)

	# dodaj współrzędne ramki granicznej i tekst OCR do listy wyników
	results.append(((startX, startY, endX, endY), text))

# posortuj wyniki współrzędnych ramki granicznej od góry do dołu
results = sorted(results, key=lambda r:r[0][1])

# loopuj po wynikach
for ((startX, startY, endX, endY), text) in results:
	# display the text OCR'd by Tesseract
	print("znaleziony tekst OCR")
	print("========")
	print("{}\n".format(text))

	# usuń tekst spoza ASCII, abyśmy mogli narysować tekst na obrazie za pomocą OpenCV, a następnie narysować tekst i ramkę otaczającą obszar tekstowy obrazu wejściowego
	text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
	output = orig.copy()
	cv2.rectangle(output, (startX, startY), (endX, endY),
		(0, 0, 255), 2)
	#font hershley simplex to wbudowany w opencv font. Jeśli ktoś chce - niech wrzuci sobie ttf'a
	#font = ImageFont.truetype("Roboto-Regular.ttf", 50)
	cv2.putText(output, text, (startX, startY - 20),
		cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 3)

	# wyswietl sprasowany obraz
	cv2.imshow("detekcja tekstu", output)
	cv2.waitKey(0)