# Analisi explorativa e modello predittivo dati Consip

Questo progetto include una analisi esplorativa delle partecipazioni a bandi della PA Italiana 
da parte di vari fornitori nel periodo compreso fra il Gennaio 2011 e Marzo 2020 
e un modello predittivo atto a predirre il numero di partecipazioni giornaliere per il periodo 
fra Gennaio 2016 e Marzo 2020

L'analisi esplorativa e' implementata nel notebook ``notebooks/PartecipazioniAnalysis.ipynb`` mentre 
le previsioni sono implementate nel notebook  ``notebooks/PartecipazioniModel.ipynb``.

Il modulo ``models/models.py`` contiene la definizione dei modelli predittivi utilizzati.

Sono richieste le seguenti labriries:

* numpy
* pandas
* sklearn
* tensorflow
* bokeh
