# ==============================================================================
# TopoWeather AI – Fáze 2: Příprava a čištění dat
# Tento skript vezme surová data z CSV, vyčistí je a připraví pro AI model.
# Hlavní úkol: škálování čísel, aby model "viděl" všechny hodnoty stejně.
# ==============================================================================

import pandas            # Knihovna pro práci s tabulkami (načítání, čištění CSV)
import os                # Knihovna pro práci se složkami a soubory

# Z knihovny scikit-learn importujeme jen nástroj MinMaxScaler
# (nechceme importovat celou knihovnu, stačí nám tento jeden nástroj)
from sklearn.preprocessing import MinMaxScaler

# ==============================================================================
# NASTAVENÍ – cesty k souborům
# ==============================================================================

VSTUPNI_SOUBOR  = "data/raw/meteo_raw.csv"
VYSTUPNI_SOUBOR = "data/processed/meteo_processed.csv"

# Seznam sloupců, které budeme škálovat
# POZOR: 'extremni_srazky' zde NESMÍ být – to je náš cíl (label), ten necháme být
SLOUPCE_PRO_SKALOAVNI = ["sirka", "delka", "nadmorska_vyska", "max_teplota"]


# ==============================================================================
# KROK 1: NAČTENÍ DAT
# Pandas přečte CSV soubor a uloží ho do proměnné jako tabulku (DataFrame)
# ==============================================================================

print("="*60)
print("KROK 1: Načítám surová data ze souboru...")
print("="*60)

tabulka = pandas.read_csv(VSTUPNI_SOUBOR)

# Vypíšeme základní info o načtené tabulce
print(f"Načteno řádků: {len(tabulka)}")
print(f"Načteno sloupců: {len(tabulka.columns)}")
print(f"Názvy sloupců: {list(tabulka.columns)}")


# ==============================================================================
# KROK 2: KONTROLA A ČIŠTĚNÍ DAT
#
# Reálná data z internetu nikdy nejsou dokonalá. Mohou obsahovat:
#   - NaN (Not a Number) = prázdná buňka, API nemělo data pro daný den
#   - Duplikáty = stejný řádek zapsaný dvakrát (chyba při stahování)
# Model strojového učení si s NaN hodnotami neporadí, musíme je odstranit.
# ==============================================================================

print("\n" + "="*60)
print("KROK 2: Kontroluji a čistím data...")
print("="*60)

# --- Prázdné hodnoty (NaN) ---

# Spočítáme, kolik prázdných hodnot je v každém sloupci
pocet_nan = tabulka.isnull().sum()
print(f"\nPočet prázdných hodnot (NaN) v každém sloupci:")
print(pocet_nan)

# Zapamatujeme si počet řádků PŘED čištěním
pocet_radku_pred = len(tabulka)

# Odstraníme všechny řádky, kde je KDEKOLIV prázdná hodnota
# (dropna = "drop NaN" = vyhoď prázdné)
tabulka = tabulka.dropna()

pocet_radku_po_nan = len(tabulka)
odstraneno_nan = pocet_radku_pred - pocet_radku_po_nan
print(f"\nOdstraněno řádků kvůli NaN: {odstraneno_nan}")


# --- Duplicitní řádky ---

# Spočítáme, kolik řádků je úplně stejných jako jiný řádek
pocet_duplikatu = tabulka.duplicated().sum()
print(f"\nPočet duplicitních řádků: {pocet_duplikatu}")

# Odstraníme duplicity – necháme vždy jen první výskyt
tabulka = tabulka.drop_duplicates()

pocet_radku_po_dup = len(tabulka)
odstraneno_dup = pocet_radku_po_nan - pocet_radku_po_dup
print(f"Odstraněno duplicitních řádků: {odstraneno_dup}")

print(f"\nVýsledný počet čistých řádků: {len(tabulka)}")


# ==============================================================================
# KROK 3: UKÁZKA HODNOT PŘED ŠKÁLOVÁNÍM
# Uložíme si 3 ukázkové řádky PŘED škálováním, abychom je mohli porovnat potom
# ==============================================================================

print("\n" + "="*60)
print("KROK 3: Ukázka hodnot PŘED škálováním (první 3 řádky):")
print("="*60)

# Nastavíme pandas, aby vypisoval všechny sloupce (ne zkrácené "...")
pandas.set_option("display.max_columns", None)
pandas.set_option("display.width", None)

# Uložíme si ukázku do samostatné proměnné pro porovnání
ukazka_pred = tabulka[SLOUPCE_PRO_SKALOAVNI].head(3).copy()
print(ukazka_pred)

print("\nJak vidíš, hodnoty jsou v úplně jiných řádech:")
print(f"  - 'sirka' je okolo 48–51 (stupně zeměpisné šířky)")
print(f"  - 'delka' je okolo 12–19 (stupně zeměpisné délky)")
print(f"  - 'nadmorska_vyska' je třeba 477 (metry!)")
print(f"  - 'max_teplota' je třeba -5 až +35 (stupně Celsia)")


# ==============================================================================
# KROK 4: ŠKÁLOVÁNÍ POMOCÍ MinMaxScaler
#
# PROČ ŠKÁLUJEME? – vysvětlení pro komisi:
#
# Představ si, že trénuješ model, který se má naučit z čísel.
# Jeden sloupec má hodnoty 12–19 (délka), jiný má 0–800 (výška v metrech).
#
# Model počítá vzdálenosti a váhy mezi hodnotami. Pokud jeden sloupec má
# hodnoty stokrát větší než jiný, model si "myslí", že ten větší sloupec
# je stokrát důležitější – jen proto, že má velká čísla!
#
# MinMaxScaler přepočítá KAŽDÝ sloupec zvlášť do rozmezí 0 až 1:
#
#   nová_hodnota = (původní - minimum) / (maximum - minimum)
#
# Příklad pro nadmořskou výšku: minimum=150m, maximum=1400m
#   - 150m  → (150-150)/(1400-150)  = 0.0   (nejnižší bod → 0)
#   - 775m  → (775-150)/(1400-150)  = 0.5   (střed → 0.5)
#   - 1400m → (1400-150)/(1400-150) = 1.0   (nejvyšší bod → 1)
#
# Po škálování jsou VŠECHNY sloupce rovnocenné a model se může férově
# naučit, který z nich skutečně předpovídá extrémní srážky.
# ==============================================================================

print("\n" + "="*60)
print("KROK 4: Škáluji data pomocí MinMaxScaler...")
print("="*60)

# Vytvoříme instanci škálovače (jako kdybys vytáhl kalkulačku z šuplíku)
skaloavac = MinMaxScaler()

# Vezmeme z tabulky jen sloupce určené ke škálování
# .values vrátí čistá čísla (numpy array) bez názvů sloupců
data_ke_skaloavni = tabulka[SLOUPCE_PRO_SKALOAVNI].values

# fit_transform udělá dvě věci najednou:
#   1) "fit"       = prohlédne si data, zjistí minimum a maximum každého sloupce
#   2) "transform" = přepočítá všechna čísla do rozmezí 0–1
data_po_skaloavni = skaloavac.fit_transform(data_ke_skaloavni)

print("Škálování dokončeno!")

# Vypíšeme, jaké minimum a maximum škálovač našel v každém sloupci
print("\nZjištěné minimální hodnoty v datech:")
index = 0
while index < len(SLOUPCE_PRO_SKALOAVNI):
    nazev_sloupce = SLOUPCE_PRO_SKALOAVNI[index]
    minimum = round(skaloavac.data_min_[index], 2)
    maximum = round(skaloavac.data_max_[index], 2)
    print(f"  {nazev_sloupce}: min={minimum}, max={maximum}")
    index = index + 1


# ==============================================================================
# KROK 5: SESTAVENÍ FINÁLNÍ TABULKY
# Přeneseme škálovaná čísla zpět do pandas tabulky se správnými názvy sloupců
# ==============================================================================

print("\n" + "="*60)
print("KROK 5: Sestavuji finální tabulku...")
print("="*60)

# Vytvoříme novou tabulku jen ze škálovaných hodnot
# (data_po_skaloavni je teď pole čísel 0–1)
tabulka_skala = pandas.DataFrame(data_po_skaloavni, columns=SLOUPCE_PRO_SKALOAVNI)

# Přidáme zpátky sloupec 'extremni_srazky' – ten jsme NEškálovali, necháme ho jako 0/1
# reset_index() zajistí, že indexy řádků sedí správně po dropna() a drop_duplicates()
tabulka_skala["extremni_srazky"] = tabulka["extremni_srazky"].reset_index(drop=True)

print(f"Finální tabulka má {len(tabulka_skala)} řádků a {len(tabulka_skala.columns)} sloupce.")


# ==============================================================================
# KROK 6: UKÁZKA HODNOT PO ŠKÁLOVÁNÍ + POROVNÁNÍ
# ==============================================================================

print("\n" + "="*60)
print("KROK 6: Ukázka hodnot PO škálování (první 3 řádky):")
print("="*60)

ukazka_po = tabulka_skala[SLOUPCE_PRO_SKALOAVNI].head(3)
print(ukazka_po)

print("\n--- SROVNÁNÍ PŘED a PO škálování ---")
print("\nPŘED škálováním:")
print(ukazka_pred.to_string(index=False))
print("\nPO škálování:")
print(ukazka_po.to_string(index=False))
print("\nVšechny hodnoty jsou nyní v rozmezí 0.0 až 1.0 – model je spokojený!")

# Ověříme rozložení cílového sloupce (kolik je jedniček a nul)
pocet_extremu  = int(tabulka_skala["extremni_srazky"].sum())
pocet_normalu  = len(tabulka_skala) - pocet_extremu
print(f"\nRozložení cílového sloupce 'extremni_srazky':")
print(f"  Extrémní dny (1): {pocet_extremu}")
print(f"  Normální dny  (0): {pocet_normalu}")


# ==============================================================================
# KROK 7: ULOŽENÍ DO CSV
# ==============================================================================

print("\n" + "="*60)
print("KROK 7: Ukládám výsledek do CSV...")
print("="*60)

# Vytvoříme složku data/processed/ pokud ještě neexistuje
vystupni_slozka = os.path.dirname(VYSTUPNI_SOUBOR)
if not os.path.exists(vystupni_slozka):
    os.makedirs(vystupni_slozka)
    print(f"Složka '{vystupni_slozka}' byla vytvořena.")

# Uložíme finální tabulku jako CSV (index=False = bez sloupce s čísly řádků)
tabulka_skala.to_csv(VYSTUPNI_SOUBOR, index=False, encoding="utf-8")

print(f"\nHotovo! Soubor uložen: {VYSTUPNI_SOUBOR}")
print(f"Celkový počet připravených záznamů: {len(tabulka_skala)}")
print("\nData jsou připravena pro Fázi 3 – trénování AI modelu!")