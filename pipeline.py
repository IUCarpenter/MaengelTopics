import argparse
import json
import re
import spacy

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

############# SETTINGS #############

JSON_PATH = "maengel.json"     # JSON Datei mit Maengelmeldungen
MIN_WORDS = 3                   # Mindestwoerter damit Mangel zaehlt
TOP_N = 10                      # Top-Woerter pro Thema/Cluster fuer Ausgabe




class MaengelJson:
    """
    Holt den Spalte mit key "description" aus JSON Datei
    """
    def __init__(self, path):
        self.path = path

    def load_descriptions(self):
        desc = []
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            try:
                desc.append(str(item["description"]))
            except:
                desc.append("")
        return desc


class Cleaning:
    """
    Regelbasierte Bereinigung der Maengelmeldungen mit RegEx
    """
    URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
    NONLETTER_RE = re.compile(r"[^a-zA-ZäöüÄÖÜß\s]")
    MULTISPACE_RE = re.compile(r"\s+")

    manual_stop = [
    "bitte", "hallo", "danke", "sorry", "entschuldigung", "links", "rechts",
    "sehr", "geehrte", "geehrter", "geehrt", "rum", "ost", "west", "bereich",
    "freundlich", "freundliche", "freundlichen", "gruß", "grüße", "gruesse",
    "mit", "mfg", "ihr", "ihre", "ihren", "ihrer", "euer", "eure", "euren", "eurer",
    "herr", "herren", "frau", "dame", "damen", "danke", "dank", "hausnummer",
    "mal", "einfach", "halt", "eben", "auch", "noch", "schon", "nur", "ja", "nee",
    "dann", "bitten", "vielleicht", "irgendwie", "max", "richtung", "einfach",
    "woche", "wochen", "tag", "tage", "monate", "seit", "immer", "heute", "gestern",
    "stadt", "jena", "monat"
    ]

    def __init__(self, min_words=3):
        self.min_words = min_words

    def run(self, texts):
        clean = []
        for t in texts:
            t = t.lower()
            t = self.URL_RE.sub(" ", t)
            t = self.NONLETTER_RE.sub(" ", t)
            t = self.MULTISPACE_RE.sub(" ", t).strip()
            t = " ".join(w for w in t.split() if w not in self.manual_stop)
            
            if len(t.split()) >= self.min_words:
                clean.append(t)
        return clean


class SpacyPreprocessor:
    """
    Erzeugen von Lemmas mit spaCy Modell: de_core_news_sm
    Ausgabe als Liste (string pro Dokument, tokens sind per split rekonstruierbar)
    """
    def __init__(self, model="de_core_news_sm", min_words=3):
        self.min_words = min_words
        self.nlp = spacy.load(model, disable=["ner", "parser", "textcat"])

    def run(self, texts):
        processed = []
        for doc in self.nlp.pipe(texts, batch_size=128):
            toks = [t.lemma_ for t in doc if not t.is_stop and t.is_alpha and len(t.text) > 2]
            if len(toks) >= self.min_words:
                processed.append(" ".join(toks))
        return processed


class Vectorizer:
    """
    Erstellen von Wort-Vektoren mit Bag-of-Words sowie TF-IDF
    """
    def __init__(self, min_df=2):
        self.min_df = min_df

    def bow(self, texts):
        v = CountVectorizer(min_df=self.min_df)
        X = v.fit_transform(texts)
        f = v.get_feature_names_out()
        return X, f

    def tfidf(self, texts):
        v = TfidfVectorizer(min_df=self.min_df)
        X = v.fit_transform(texts)
        f = v.get_feature_names_out()
        return X, f


class Extractor:
    """
    Themenextraktion mit LSA und K-Means
    """
    def _top_words(self, weights, features, top_n):
        top = weights.argsort()[::-1][:top_n]
        return [features[i] for i in top]

    def lsa_topics(self, X, features, k, top_n):
        lsa = TruncatedSVD(n_components=k, random_state=42)
        lsa.fit(X)
        return [self._top_words(comp, features, top_n) for comp in lsa.components_]

    def kmeans_topics(self, X, features, k, top_n):
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        km.fit(X)
        return [self._top_words(center, features, top_n) for center in km.cluster_centers_]


class Coherence:
    """
    Coherence Score (cv = coherence Value) berechnen mit gensim
    """
    
    def __init__(self, processed_texts):
        self.tokens_list = [doc.split() for doc in processed_texts]
        self.dictionary = Dictionary(self.tokens_list)
        self.dictionary.filter_extremes(no_below=5, no_above=0.4)

    def cv(self, topics):
        cmodel = CoherenceModel(topics=topics, texts=self.tokens_list, dictionary=self.dictionary, coherence="c_v")
        coh = cmodel.get_coherence()
        return coh


class Pipeline:
    """
    Orchestrierung, Programmablauf
    """
    def __init__(self):
        self.loader = MaengelJson(JSON_PATH)
        self.cleaning = Cleaning(MIN_WORDS)
        self.spacy = SpacyPreprocessor(min_words=MIN_WORDS)
        self.vectorizer = Vectorizer(min_df=2)
        self.extractor = Extractor()

    def run(self, k, do_coh):
        print("its Running... Bitte warten.. (Theres a lot to process)")
        raw = self.loader.load_descriptions()
        print("Cleanjob...")
        cleaned = self.cleaning.run(raw)
        print("Lemmatisierung...")
        processed = self.spacy.run(cleaned)

        print("Einträge insgesamt:", len(raw))
        print("Einträge nach Cleaning:", len(cleaned))
        print("Einträge nach spaCy:", len(processed))

        print("Vektorisierung...")
        X_bow, bow_features = self.vectorizer.bow(processed)
        X_tfidf, tfidf_features = self.vectorizer.tfidf(processed)

        print("BoW Matrix Form    :", X_bow.shape)
        print("TF-IDF Matrix Form :", X_tfidf.shape)

        print("Themen werden extrahiert...")
        lsa_tfidf = self.extractor.lsa_topics(X_tfidf, tfidf_features, k, TOP_N)
        km_tfidf  = self.extractor.kmeans_topics(X_tfidf, tfidf_features, k, TOP_N)

    
        lsa_bow = self.extractor.lsa_topics(X_bow, bow_features, k, TOP_N)
        km_bow  = self.extractor.kmeans_topics(X_bow, bow_features, k, TOP_N)


        print("\n#############################################################")
        print(f"LSA (TF-IDF) Themen (k = {k}) - Top {TOP_N}")
        print("############################################################")
        for i, words in enumerate(lsa_tfidf, start=1):
            print(f"Thema {i:02d}: " + ", ".join(words))

        print("\n############################################################")
        print(f"LSA (BoW) Themen (k = {k}) - Top {TOP_N}")
        print("############################################################")
        for i, words in enumerate(lsa_bow, start=1):
            print(f"Thema {i:02d}: " + ", ".join(words))




        print("\n############################################################")
        print(f"KMeans (TF-IDF) Cluster (k = {k}) - Top {TOP_N}")
        print("##############################################################")
        for i, words in enumerate(km_tfidf, start=1):
            print(f"Cluster {i:02d}: " + ", ".join(words))

        print("\n##############################################################")
        print(f"KMeans (BoW) Cluster (k = {k}) - Top {TOP_N}")
        print("############################################################")
        for i, words in enumerate(km_bow, start=1):
            print(f"Cluster {i:02d}: " + ", ".join(words))



############# COHERENCE FLOW START #############

        if do_coh:
            print("############################################################")
            print("Coherence Score wird berechnet...")
            print("Heavy Work - kann einige Minuten dauern")
            coh = Coherence(processed)
            s_lsa_tfidf = coh.cv(lsa_tfidf)
            s_km_tfidf  = coh.cv(km_tfidf)
            s_lsa_bow   = coh.cv(lsa_bow)
            s_km_bow    = coh.cv(km_bow)

            print("\n############################################################")
            print("Coherence Score")
            print("############################################################")
            print(f"k={k}")
            print(f"LSA   TF-IDF : {s_lsa_tfidf:.4f}")
            print(f"KMeans TF-IDF: {s_km_tfidf:.4f}")
            print(f"LSA   BoW    : {s_lsa_bow:.4f}")
            print(f"KMeans BoW   : {s_km_bow:.4f}")

############# COHERENCE FLOW END #############

def parse_args():
    """
    Helperfunktion nimmt Argumente aus CLI an
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, required=True, help="Anzahl Themen/Cluster (k)")
    ap.add_argument("-coh", action="store_true", help="berechne zusätzlich Coherence")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Pipeline().run(k=args.k, do_coh=args.coh)
