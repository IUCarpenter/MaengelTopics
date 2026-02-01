KachelDash - Projektarbeit für IU


Dieses Projekt implementiert eine NLP-Pipeline in Python. Die Aufgabenstellung stammt aus einem IU-Projekt. Hierbei handelt es sich um den praktischen Teil aus Phase 2 (Reflexionsphase).



Ziel: Ein Datensatz mit Einträgen aus einem Mängelmelde-Formular der Stadt Jena wird mithilfe von Kombinationen aus BoW, TF-IDF, LSA und k-Means ausgewertet.
Die Coherence Score der einzelnen Kombinationen wird ausgegeben.

Diese README-Datei dient als INSTALLATIONSANLEITUNG



Voraussetzungen für beide Methoden:
Python 3.10+ muss auf dem System installiert sein!!
Projektdateien müssen lokal im Projektverzeichnis liegen

Es gibt zwei Methoden für die Installation des Programms unter Windows:

1. ######### Automatisch #########
- Projektverzeichnis im Explorer öffnen

- "01-installer.bat" durch Doppeklick ausführen
	...warten.

- "02-run.bat" durch Doppeklick ausführen
	User Input promps beantworten.
	...warten.



2. ######### Manuell #########
- CLI im Projektverzeichnis öffnen.

- Venv erzeugen: 
"python -m venv nlpEnv"

- Venv aktivieren: 
"nlpEnv\scripts\activate"

- Requirements und spaCy Sprachmodell für Lemmatisierung installieren:
"pip install -r requirements.txt"
"python -m spacy download de_core_news_sm"

- Programm ausführen:
"python Pipeline.py -k x -coh"
Hierbei wird "x" durch die gewünschten Themenanzahl ersetzt.
Das Startargument "-coh" ist optional und aktiviert die Berechnung
der Coherence Score.























Verwendeter Datensatz: 
Jena.OpenData. (2018, September 4). Im Mängelmelder gemeldete Mängel. maengel.json. https://opendata.jena.de/dataset/mangel/resource/86e7e63a-696f-49af-b284-8d9c5069da35
