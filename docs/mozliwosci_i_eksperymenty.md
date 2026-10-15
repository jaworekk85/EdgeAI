# Co można zbadać przy pomocy EdgeAI?

Stan na 15 października 2026 r. Przewodnik po możliwościach i pomysłach na badania.
Opisane eksperymenty są propozycjami do wyboru; ten dokument ich nie uruchamia.

**Repo pomaga odpowiedzieć na pytanie: jaki wariant modelu daje wystarczającą
jakość przy najmniejszym koszcie obliczeń?** Dziś mierzymy ten koszt na CPU jako
czas, zużycie pamięci i rozmiar pliku. Po wdrożeniu obsługi Jetsona dojdą pomiary
mocy, energii i temperatury. Kolejne fazy pozwolą badać automatyczny wybór modelu
oraz jego zachowanie przy pogarszających się warunkach.

## 1. Co jest dostępne teraz?

Sprawdzony przykład to **klasyfikacja obrazów**: MobileNetV3 Small i Large,
PyTorch i ONNX Runtime na CPU, obrazy z Imagenette. Model wybiera jedną z 1000
klas ImageNet. Standardowy test wykorzystuje 20 obrazów, po dwa na klasę Imagenette.
Inferencja oznacza tutaj uruchomienie modelu dla danego wejścia; aktywacje to
sygnały pośrednie wewnątrz sieci.

| Możliwość | Co faktycznie otrzymujemy |
| --- | --- |
| Porównanie Small i Large | Trafność, czas obliczeń, przepustowość i pamięć dla dwóch wielkości modelu |
| Porównanie PyTorch i ONNX Runtime | Koszt uruchamiania tego samego modelu przez dwa różne środowiska wykonania |
| Kompresja modelu | FP16, INT8, częściowy INT4 i trzy rodzaje masek pruningowych; sprawdzona macierz korzysta z MobileNetV3 Small |
| Diagnostyka INT8 | Porównanie wyników, wag i aktywacji; wskazanie miejsc, w których kwantyzacja powoduje duży błąd |
| Dłuższe działanie na CPU | Powtarzane pełne przebiegi, rzeczywista długość testu i zmiany opóźnienia w czasie |
| Raport i odtwarzalność | Tabele, wykresy, surowe czasy, konfiguracja, wersje bibliotek, identyfikatory modeli i obrazów; każdy przebieg zapisany osobno |

Mierzymy średnie opóźnienie, medianę, p95/p99, FPS, rozmiar modelu i próbkowany
szczyt pamięci procesu. **p95** oznacza czas, którego nie przekracza 95% obserwacji;
jest przydatny, gdy sporadyczne opóźnienia przeszkadzają bardziej niż średnia.
Pamięć procesu obejmuje także biblioteki i środowisko wykonania.

Ograniczenia obecnych metod kompresji:

- **FP16:** zmniejsza precyzję modelu i wejścia w PyTorch. Na tym CPU był wolniejszy.
- **INT8:** obejmuje wagi i aktywacje części operacji. Domyślna szeroka receptura
  mocno traci jakość; jest już wariant mieszany `linear`, zachowujący sploty w FP32.
- **INT4:** zmienia tylko wagi końcowego klasyfikatora; sploty i aktywacje pozostają FP32.
- **Pruning:** zeruje wagi, ale zachowuje gęste tensory i ich rozmiary. Dotyczy to
  także masek całych kanałów i wzorca 2:4. Fizycznego usuwania kanałów jeszcze nie ma.
  Unstructured zeruje pojedyncze wagi, channel wagi wybranych kanałów, a 2:4
  pozostawia najwyżej dwie niezerowe wagi w każdej obsługiwanej grupie czterech.

Energia w obecnych raportach pochodzi z założonej mocy. Jest **oszacowaniem**,
a nie pomiarem zużycia prądu przez komputer. Odtwarzanie logów telemetrycznych
służy sprawdzaniu obsługi danych; nie symuluje wydajności przyszłego Jetsona.

## 2. Konkretne eksperymenty możliwe już na CPU

Poniższe porównania wykorzystują istniejące narzędzia. Zmiana konfiguracji lub
doboru danych może wymagać nowego przygotowania modeli i manifestów obrazów.
Repo nie uruchamia automatycznego przeszukiwania wszystkich kombinacji.

### A. Ile kosztuje większy model i ile daje jakości?

**Pytanie:** czy MobileNetV3 Large daje wystarczającą poprawę względem Small,
aby uzasadnić dodatkowy czas i pamięć?

**Porównanie:** Small i Large w FP32, na tych samych obrazach, osobno w PyTorch
i ONNX Runtime. Następnie można porównać ten sam model między środowiskami,
z jawnym dopuszczeniem różnicy środowiska w raporcie.

**Pomiary:** trafność, średni czas i p95, FPS, rozmiar pliku, szczyt pamięci.
**Wynik:** wybór wariantu dla zadanego limitu czasu lub pamięci; oddzielenie wpływu
architektury modelu od wpływu biblioteki, która go wykonuje.

### B. Czy mniejszy plik oznacza szybszą inferencję?

**Pytanie:** które dostępne kompresje przynoszą użyteczną korzyść na naszym CPU?

**Porównanie:** FP32–FP16 w PyTorch oraz FP32–INT8–INT4 w ONNX Runtime.
Oceniać trzeba również zakres zmiany: obecne INT4 i INT8 obejmują różne części sieci.

**Pomiary:** trafność, czas, pamięć i rozmiar pliku. Każdy wariant porównujemy
z FP32 w tym samym środowisku wykonania.
**Wynik:** wskazanie wariantów spełniających budżet jakości i zasobów. Zmniejszenie
pliku przy jednoczesnym spowolnieniu też jest wartościowym wynikiem.

### C. Jak kalibracja wpływa na INT8?

**Pytanie:** jak zmienia się jakość przy zwiększeniu liczby obrazów użytych do
wyznaczenia zakresów kwantyzacji?

**Porównanie:** obecne 10 obrazów kalibracyjnych i większy, wcześniej ustalony
podzbiór treningowy, np. 100 obrazów. Parametr liczby obrazów na klasę istnieje
w konfiguracji. Zbiór oceny pozostaje ten sam i nie uczestniczy w kalibracji.

**Pomiary:** trafność i zmiany przewidywań; do dokładniejszej analizy rozbieżności
warstw służy diagnostyka. Jej obecny skrypt obsługuje maksymalnie 64 obrazy na
podzbiór, więc nie przeanalizuje bez zmian całej kalibracji liczącej 100 obrazów.
**Wynik:** sprawdzenie wpływu liczby danych, bez zakładania, że większa kalibracja
rozwiąże problem. Znamy już utratę małych sygnałów przy wspólnej skali aktywacji.

Wykonano już [sprawdzenie wariantu mieszanego](int8_diagnosis.md): 66% trafności
wobec 66,5% FP32 na 200 nowych obrazach, przy 100 obrazach kalibracyjnych.
Ustawienie `int8_scope: linear` zachowuje sploty w FP32. Szersza kwantyzacja nadal
traci jakość; inne zakresy lub metody kalibracji wymagają dalszego rozszerzenia.

### D. Jak szybko pruning niszczy jakość?

**Pytanie:** jak trafność reaguje na wzrost liczby wyzerowanych wag?

**Porównanie:** wcześniej wybrane poziomy masek, np. 10%, 25% i 50% dla metod
unstructured i channel, oraz osobny wariant 2:4. Dla 2:4 proporcja wynika ze wzorca;
nie jest to swobodnie regulowany poziom przycinania.

**Pomiary:** trafność, rzeczywista liczba zer, czas, pamięć i rozmiar pliku.
**Wynik:** krzywa utraty jakości oraz sprawdzenie, czy zera dają jakikolwiek zysk
przy obecnym gęstym wykonaniu. Sam wzrost liczby zer nie oznacza mniej operacji
ani mniejszego pliku. Odzyskiwanie jakości przez dotrenowanie to dalsze rozszerzenie.

### E. Czy wydajność utrzymuje się podczas dłuższej pracy?

**Pytanie:** czy opóźnienia na CPU zmieniają się po dłuższym wykonywaniu modelu?

**Porównanie:** ten sam model i protokół w kilku niezależnych dłuższych przebiegach;
analiza początkowych i końcowych okien pomiarowych każdego przebiegu.

**Pomiary:** rozkład czasów, p95, zmiana średniego opóźnienia, szczyt pamięci
i rzeczywista liczba inferencji. Trzeba kontrolować inne obciążenia komputera.
**Wynik:** opis stabilności na tym CPU. Same zmiany czasu nie dowodzą przegrzewania.
Powtarzanie tych samych obrazów pomaga badać czas, ale nie powiększa próby jakości.

## 3. Co odblokują kolejne planowane fazy?

To kolejność rozwoju funkcji, a nie obietnica dat. Każda faza wymaga osobnego
przeglądu. Fazy 5 i 6 można rozwijać na CPU przed zakupem Jetsona; 3B rozpoczyna
się po uzyskaniu sprzętu i zatwierdzeniu jego obsługi.

| Etap | Nowa możliwość | Przykładowe badanie |
| --- | --- | --- |
| **Faza 5 — adaptacyjne obliczenia** | Wybór modelu dla kolejnych obrazów, przetwarzanie sekwencji, pomiar kosztów ładowania i przełączania | Czy mały model na łatwych obrazach i duży na trudnych daje lepszy kompromis niż jeden stały wariant? |
| **Faza 6 — zmienne warunki** | Powtarzalne zmiany jasności, rozmycia, szumu i ich harmonogramy; sygnały do oceny wystarczalności modelu | Kiedy model skompresowany przestaje wystarczać i czy potrafimy to wykryć? |
| **Faza 3B — sprzęt** | Backend TensorRT, obserwacje urządzenia, bieżąca telemetria, rzeczywista moc i temperatura | Które kompresje i strategie wyboru zmniejszają energię przy zachowaniu jakości i limitu opóźnienia? |

### F. Czy automatyczny wybór modelu się opłaca? — po fazie 5

Porównać trzy strategie: zawsze Small, zawsze Large oraz wybór na podstawie
pewności predykcji. Dodać **oracle** jako punkt odniesienia: analizę offline,
która zna poprawne odpowiedzi i sprawdza, co dałby idealny wybór spośród wariantów.
Oracle nie jest strategią dostępną podczas rzeczywistej pracy.

Mierzyć trafność, p95, udział użycia każdego modelu, pamięć oraz koszt samego wyboru
i przełączania. Uwzględnić sytuację, gdy oba modele są w pamięci, i sytuację
wymagającą przeładowania. Na CPU energia nadal będzie szacowana.

**Interesujący wynik:** oszczędność obliczeń utrzymuje się po doliczeniu narzutów.
Może też się okazać, że częste przełączanie pochłania cały zysk. Duża pewność modelu
nie gwarantuje poprawnej odpowiedzi — to również trzeba ocenić.

### G. Jak kompresja wpływa na odporność? — po fazie 6

Porównać FP32 i wybrane kompresje na tych samych obrazach przy kilku ustalonych
poziomach rozmycia, szumu i jasności. Następnie stworzyć sekwencję, w której warunki
stopniowo się pogarszają i wracają do normy, i zbadać reakcję kontrolera z fazy 5.

Mierzyć spadek trafności, p95, liczbę błędnych odpowiedzi z dużą pewnością,
moment przełączenia oraz zbędne przełączenia tam i z powrotem.
**Wynik:** wskazanie, kiedy potrzebny jest mocniejszy wariant i czy dostępne sygnały
ostrzegają dostatecznie wcześnie. Syntetyczne rozmycie czy zmiana jasności badają
konkretne zaburzenie; nie zastępują testów z rzeczywistej kamery w terenie.

### H. Co naprawdę daje kompresja na Jetsonie? — po fazie 3B

Przygotować i zweryfikować warianty dla rzeczywistego TensorRT oraz porównać je
na tym samym urządzeniu, przy zapisanych ustawieniach i jednakowym protokole.
Powtórzyć wybrane porównania z CPU, a po fazie 5 także porównania kontrolerów.
Obecnych plików INT4 dla ONNX Runtime nie traktujemy jako gotowych silników TensorRT.
Obsługę każdej precyzji i sparse kernels trzeba sprawdzić na docelowym stosie.

Mierzyć trafność, czas, pamięć, moc, energię na obraz uśrednioną w dłuższym oknie
oraz temperaturę. Sprawdzać, które operacje faktycznie używają niskiej precyzji
lub rzadkości 2:4. Parametry/FLOPs i liczba zer pozostają opisem modelu, nie dowodem
przyspieszenia. Osobnym porównaniem może być wpływ trybu zasilania urządzenia.

**Wynik:** wybór wariantu lub strategii, która spełnia budżet jakości i opóźnienia
przy najmniejszej **zmierzonej** energii. Wyników czasu z CPU nie przeliczamy
na przewidywaną szybkość Jetsona.

## 4. Jak zamienić sprawdzenie pipeline'u w sensowne badanie?

Przed uruchomieniem ustalić:

1. **Jedno pytanie i kryterium sukcesu.** Na przykład dopuszczalny spadek trafności
   i maksymalne p95. Progi wybieramy przed oceną końcową.
2. **Większy, reprezentatywny zbiór.** Obecne 20 obrazów sprawdza działanie kodu.
   Jedna odpowiedź zmienia trafność aż o 5 punktów procentowych. Powiększenie
   podzbioru Imagenette wymaga zgodnego ustawienia liczby iteracji i nowego przygotowania.
3. **Rozdzielenie danych.** Kalibracja, dobieranie ustawień i końcowa ocena powinny
   korzystać z rozdzielonych próbek. Wielokrotne wybieranie najlepszej receptury
   na tym samym zbiorze oceny osłabia wiarygodność wyniku.
4. **Porównywalny protokół.** Te same obrazy, preprocessing, liczba wątków, warm-up
   i warunki wykonania; zmieniamy jasno wskazany czynnik. Różnice środowiska lub
   protokołu jawnie oznaczamy. Przy zmianie samego środowiska pilnujemy parytetu modelu.
5. **Powtórzenia i niepewność.** Więcej pomiarów czasu dla p95/p99, niezależne
   powtórzenia przebiegów oraz ocena niepewności trafności. Repo przechowuje wyniki;
   pełna analiza statystyczna nie jest obecnie automatycznym raportem.

Dotychczasowy wynik Small na 20 obrazach to 70% dla FP32/FP16/INT4, 5% dla INT8,
40% dla unstructured i 0% dla channel/2:4. To obserwacje tych konkretnych receptur,
które wskazały problemy do zbadania. Nie są rankingiem metod kompresji.

## 5. Proponowana kolejność prac

Najpierw **A i B na większym zbiorze**, aby ustalić punkt odniesienia oraz budżet
jakości. Wstępny krok obejmujący Small i mieszany INT8 na 200 obrazach już wykonano.
Dalsze **C** pozostaje osobnym badaniem; można teraz zaprojektować fazę 5
ze sprawdzonymi wariantami. Faza 5 pozwoli porównać strategie **F**,
a faza 6 ich odporność w **G**. Po uzyskaniu sprzętu 3B odblokuje **H** niezależnie
od tego, jak daleko zajdą prace CPU. **D i E** można wybrać jako osobne badania.

Detekcja obiektów, segmentacja, własne zadanie inspekcyjne, małe modele językowe
czy trening/QAT to możliwe osobne rozszerzenia. Wymagają odpowiednich modeli,
adapterów danych i metryk; nie są gotowymi funkcjami ani zobowiązaniem faz 5–6.
Pierwsza wersja kontrolera może operować na sekwencji obrazów. Wejście z kamery,
dekodowanie wideo czy integracja z robotem wymagają własnych adapterów.

Instrukcje wykonania: [CPU i przygotowanie danych](cpu_reference.md),
[kompresja](phase_4_usage.md), [diagnostyka INT8](int8_diagnosis.md),
[dłuższe przebiegi](phase_3a_usage.md). Bieżące ograniczenia i przegląd kolejnej
fazy są w [STATUS.md](../STATUS.md); wymagania sprzętowe w
[planie weryfikacji Jetsona](hardware_setup.md).
