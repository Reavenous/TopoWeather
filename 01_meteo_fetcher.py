# ==============================================================================
# TopoWeather AI – Fáze 1: Sběr meteorologických dat
# Skript stahuje historická data z bezplatného Open-Meteo Archive API
# a ukládá je do CSV souboru pro pozdější trénování modelu.
# ==============================================================================

import requests   # Knihovna pro volání webových API (stahování dat z internetu)
import pandas     # Knihovna pro práci s tabulkami a ukládání do CSV
import time       # Knihovna pro čekání mezi požadavky (aby nás API nezablokovalo)
import os         # Knihovna pro práci se soubory a složkami
import math       # Knihovna pro matematické operace (zaokrouhlování)

# ==============================================================================
# NASTAVENÍ – všechny důležité hodnoty jsou tady pohromadě, ať je snadno najdeš
# ==============================================================================

# Hranice České republiky jako "bounding box" (obdélník na mapě)
LAT_MIN = 48.6   # Nejjižnější bod ČR (zeměpisná šířka)
LAT_MAX = 51.0   # Nejsevernější bod ČR
LON_MIN = 12.1   # Nejzápadnější bod ČR (zeměpisná délka)
LON_MAX = 18.8   # Nejvýchodnější bod ČR

# Krok mřížky – každých 0.5 stupně vytvoříme jeden bod měření
KROK = 0.5

# Časové rozmezí pro stahování historických dat
DATUM_OD = "2023-01-01"
DATUM_DO = "2024-12-31"

# Práh pro "extrémní srážky" – více než 20 mm za den = extrém
PRAH_EXTREMU_MM = 20

# Každý kolikátý normální den (bez extrému) uložíme do datasetu
# Tím udržíme dataset vyvážený – nebudeme mít 99 % nul
UKLADEJ_KAZDY_N_TY_NORMAL = 25

# Cesta k výstupnímu CSV souboru
VYSTUPNI_SOUBOR = "data/raw/meteo_raw.csv"

# Adresa Open-Meteo Archive API
API_URL = "https://archive-api.open-meteo.com/v1/archive"


# ==============================================================================
# PŘÍPRAVA VÝSTUPNÍ SLOŽKY
# Pokud složka data/raw/ ještě neexistuje, vytvoříme ji automaticky
# ==============================================================================

vystupni_slozka = os.path.dirname(VYSTUPNI_SOUBOR)
if not os.path.exists(vystupni_slozka):
    os.makedirs(vystupni_slozka)
    print(f"Složka '{vystupni_slozka}' byla vytvořena.")


# ==============================================================================
# HLAVNÍ SEZNAM – sem budeme průběžně přidávat každý zpracovaný řádek dat
# Na konci ho celý uložíme do CSV
# ==============================================================================

vsechna_data = []   # Prázdný seznam, který se bude postupně plnit záznamy


# ==============================================================================
# GRID SAMPLING – procházíme mapu ČR jako mřížku (grid)
# Vnější while cyklus = pohybujeme se ze severu na jih (latitude / zeměpisná šířka)
# Vnitřní while cyklus = pohybujeme se od západu na východ (longitude / délka)
# ==============================================================================

# Počítadlo bodů – pro přehledný výpis do konzole
pocet_bodu_celkem = 0
pocet_bodu_ok = 0
pocet_bodu_chyba = 0

# Začínáme na nejjižnějším bodě a postupujeme nahoru
aktualni_lat = LAT_MIN

while aktualni_lat <= LAT_MAX:

    # Začínáme na nejzápadnějším bodě a postupujeme doprava
    aktualni_lon = LON_MIN

    while aktualni_lon <= LON_MAX:

        pocet_bodu_celkem = pocet_bodu_celkem + 1

        # Zaokrouhlíme souřadnice na 1 desetinné místo – čistší výpis a URL
        lat_zaokrouhlena = round(aktualni_lat, 1)
        lon_zaokrouhlena = round(aktualni_lon, 1)

        print(f"--- Zpracovávám bod č. {pocet_bodu_celkem}: šířka={lat_zaokrouhlena}, délka={lon_zaokrouhlena} ---")

        # ----------------------------------------------------------------------
        # SESTAVENÍ POŽADAVKU NA API
        # Definujeme, jaká data chceme stáhnout pro tento konkrétní bod mapy
        # ----------------------------------------------------------------------

        # Parametry požadavku – posíláme je jako slovník (dictionary)
        parametry_api = {
            "latitude": lat_zaokrouhlena,
            "longitude": lon_zaokrouhlena,
            "start_date": DATUM_OD,
            "end_date": DATUM_DO,
            "daily": "precipitation_sum,temperature_2m_max",  # Chceme denní srážky a max. teplotu
            "timezone": "Europe/Prague",                       # Časová zóna ČR
            "models": "best_match"                             # Necháme API vybrat nejlepší model
        }

        # ----------------------------------------------------------------------
        # VOLÁNÍ API – posíláme požadavek na internet a čekáme na odpověď
        # Pokud API odpoví chybou, skript to zachytí a přeskočí na další bod
        # ----------------------------------------------------------------------

        try:
            # Pošleme GET požadavek na API (jako kdybys v prohlížeči zadal URL)
            odpoved = requests.get(API_URL, params=parametry_api, timeout=30)

            # Zkontrolujeme, zda API odpovědělo úspěchem (HTTP kód 200)
            # Pokud ne, vyvolá se výjimka a skočíme do bloku 'except'
            odpoved.raise_for_status()

            # Převedeme odpověď z formátu JSON do Python slovníku
            json_data = odpoved.json()

        except Exception as chyba:
            # Pokud cokoliv selhalo (výpadek internetu, chyba API, timeout...)
            print(f"  CHYBA při stahování bodu [{lat_zaokrouhlena}, {lon_zaokrouhlena}]: {chyba}")
            pocet_bodu_chyba = pocet_bodu_chyba + 1

            # Přeskočíme na další bod – pokračujeme dalším krokem vnitřního cyklu
            aktualni_lon = aktualni_lon + KROK
            continue

        # ----------------------------------------------------------------------
        # VYZVEDNUTÍ NADMOŘSKÉ VÝŠKY
        # API vrací elevaci přímo v hlavní části odpovědi (ne v 'daily')
        # ----------------------------------------------------------------------

        nadmorska_vyska = json_data.get("elevation", 0)

        # ----------------------------------------------------------------------
        # VYZVEDNUTÍ DENNÍCH DAT
        # API vrací seznam dat, srážek a teplot – každý index odpovídá jednomu dni
        # ----------------------------------------------------------------------

        denni_data = json_data.get("daily", {})

        seznam_dat      = denni_data.get("time", [])               # Např. ["2023-01-01", "2023-01-02", ...]
        seznam_srazek   = denni_data.get("precipitation_sum", [])  # Srážky v mm pro každý den
        seznam_teplot   = denni_data.get("temperature_2m_max", []) # Max. teplota ve °C pro každý den

        # Pokud API vrátilo prázdné seznamy, přeskočíme tento bod
        if len(seznam_dat) == 0:
            print(f"  Bod [{lat_zaokrouhlena}, {lon_zaokrouhlena}] – API vrátilo prázdná data, přeskakuji.")
            aktualni_lon = aktualni_lon + KROK
            continue

        # ----------------------------------------------------------------------
        # PROCHÁZENÍ DNÍ – pro každý den rozhodneme, zda jde o extrém nebo ne
        # A pak ho uložíme nebo přeskočíme podle naší logiky vzorkování
        # ----------------------------------------------------------------------

        # Počítadlo normálních dnů pro tento bod – slouží k ukládání každého 25. normálního dne
        pocitadlo_normalnich = 0

        # Projdeme všechny dny pomocí jejich indexu (0, 1, 2, ...)
        index_dne = 0
        while index_dne < len(seznam_dat):

            datum        = seznam_dat[index_dne]
            srazky_mm    = seznam_srazek[index_dne]
            teplota_max  = seznam_teplot[index_dne]

            # Pokud API vrátilo None (chybějící měření), přeskočíme tento den
            if srazky_mm is None or teplota_max is None:
                index_dne = index_dne + 1
                continue

            # ------------------------------------------------------------------
            # KLASIFIKACE DNE – extrém nebo normál?
            # ------------------------------------------------------------------

            if srazky_mm > PRAH_EXTREMU_MM:
                # Den s extrémními srážkami – VŽDY ho uložíme
                extremni_srazky = 1

                # Vytvoříme slovník s jedním řádkem dat pro CSV
                radek = {
                    "sirka": lat_zaokrouhlena,
                    "delka": lon_zaokrouhlena,
                    "nadmorska_vyska": nadmorska_vyska,
                    "max_teplota": teplota_max,
                    "extremni_srazky": extremni_srazky
                }

                vsechna_data.append(radek)

            else:
                # Normální den (srážky pod prahem)
                extremni_srazky = 0
                pocitadlo_normalnich = pocitadlo_normalnich + 1

                # Uložíme jen každý 25. normální den – tím vyvážíme dataset
                if pocitadlo_normalnich % UKLADEJ_KAZDY_N_TY_NORMAL == 0:
                    radek = {
                        "sirka": lat_zaokrouhlena,
                        "delka": lon_zaokrouhlena,
                        "nadmorska_vyska": nadmorska_vyska,
                        "max_teplota": teplota_max,
                        "extremni_srazky": extremni_srazky
                    }

                    vsechna_data.append(radek)

            # Posuneme se na další den
            index_dne = index_dne + 1

        # Bod byl úspěšně zpracován
        print(f"  OK – zatím máme celkem {len(vsechna_data)} záznamů v datasetu.")
        pocet_bodu_ok = pocet_bodu_ok + 1

        # ----------------------------------------------------------------------
        # ZDVOŘILOSTNÍ PAUZA – počkáme 0.5 sekundy, aby nás API nezablokovalo
        # ----------------------------------------------------------------------

        time.sleep(0.5)

        # Posuneme se o krok na východ (další sloupec mřížky)
        aktualni_lon = aktualni_lon + KROK

    # Posuneme se o krok na sever (další řada mřížky)
    aktualni_lat = aktualni_lat + KROK


# ==============================================================================
# ULOŽENÍ DAT DO CSV
# Celý seznam slovníků převedeme na pandas tabulku a uložíme jako CSV soubor
# ==============================================================================

print("\n" + "="*60)
print("Stahování dokončeno! Ukládám data do CSV...")

# Zkontrolujeme, zda máme vůbec nějaká data k uložení
if len(vsechna_data) == 0:
    print("VAROVÁNÍ: Žádná data nebyla stažena. Zkontroluj připojení k internetu.")
else:
    # Vytvoříme pandas DataFrame (tabulku) ze seznamu slovníků
    tabulka = pandas.DataFrame(vsechna_data)

    # Uložíme tabulku do CSV souboru (index=False = nechceme sloupec s čísly řádků)
    tabulka.to_csv(VYSTUPNI_SOUBOR, index=False, encoding="utf-8")

    # Závěrečný výpis se statistikami
    print(f"Soubor uložen: {VYSTUPNI_SOUBOR}")
    print(f"Celkový počet záznamů: {len(vsechna_data)}")
    print(f"Úspěšně zpracovaných bodů mřížky: {pocet_bodu_ok}")
    print(f"Bodů s chybou (přeskočeno):        {pocet_bodu_chyba}")
    print(f"Bodů celkem:                        {pocet_bodu_celkem}")
    print("\nNáhled prvních 5 řádků datasetu:")
    print(tabulka.head())