# 🚀 Evren-Code

[![Python Sürümü](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Lisans: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Mimari: Multi-Agent | AHEE](https://img.shields.io/badge/Mimari-Multi--Agent%20%7C%20AHEE-orange.svg)](#-temel-özellikler)
[![MCP Desteği](https://img.shields.io/badge/MCP-Uyumlu-purple.svg)](https://modelcontextprotocol.io/)

**Evren-Code**, **EVREN Yapay Zeka Platformu ([https://evren.ssyz.org.tr/](https://evren.ssyz.org.tr/))** üzerinde yer alan LLM API'lerini yazılım geliştirme süreçlerinde en verimli ve etkili şekilde kullanabilmek için geliştirilmiş **terminal tabanlı akıllı bir kodlama aracı ve otonom yazılım mühendisi ajanıdır**.

Sadece soru-cevap yapan bir sohbet arayüzü değil; doğrudan projenizin dizininde çalışan, dosyalarınızı okuyan, semantik kod haritası çıkaran, cerrahi kod değişiklikleri (Unified Diff) yapan, testleri çalıştıran ve hataları kendi kendine onaran uçtan uca bir geliştirici asistanıdır.

---

## ✨ Temel Özellikler

* **Tam Entegre EVREN API Yönetimi:** Harici hiçbir betiğe ihtiyaç duymadan `evren_llm_` anahtar doğrulamasını, ilk kullanımdaki kullanım şartları onayını (`/v1/terms/accept`) ve canlı model keşfini (`/v1/models`) otomatik olarak yönetir.
* **Görev Bazlı Çoklu Model Yönlendirme (Multi-Model Router):**
  * `deepseek-v4.1-flash` (**Varsayılan Orchestrator, Architect, Planner & Reviewer**): 1M bağlam penceresi ve derin muhakeme (`<think>`) yeteneği ile genel koordinasyon, mimari tasarım, kök neden analizi ve kod incelemesi.
  * `qwen3.8-flash-next` (**Coder, Developer & Tester**): Yüksek hızlı cerrahi kod üretimi, dosya düzenleme ve birim testi yazımı.
  * `glm-5.3` (**Genel Sohbet & Araç Çağırma**): 512K bağlam penceresi ile çok adımlı görev yürütme.
  * `qwen3-vl-30b` (**Vision**): Ekran görüntüsü ve görsel arayüz (UI) hatası analizi (`analyze_image`).
  * `gemma-4-31b` (**Compactor**): Uzun oturumlarda konuşma geçmişini özetleyerek bağlam (token) optimizasyonu.
* **Hibrit Araç Çağırma Motoru:** Hem standart OpenAI `tool_calls` şemasını hem de modellerin metin içinde ürettiği `<tool_call>` / JSON bloklarını yakalar; bozuk JSON argümanlarını otomatik onarır.
* **Adaptif Hibrit Yürütme (AHEE) & Çoklu Ajan Koordinatörü:** Basit görevleri tek ajanla doğrudan çözerken, kapsamlı görevleri bağımlılık çizgesine (DAG) ayırarak dosya kilitleme güvencesiyle paralel uzman alt-ajanlara dağıtır.
* **AST Sembol Çizgesi & Proje Zekası:** Python, TypeScript/JavaScript, Java, Go ve Rust projelerinde sembolleri indeksler ve her dosya düzenlemesinde anlık sözdizimi (AST/LSP) kontrolü yapar.
* **Cerrahi Diff Önizleme & Anında Geri Alma (`/undo`):** Dosyaları değiştirmeden önce yapılan ekleme ve çıkarmaları renkli olarak (`+` / `-`) gösterir, onayınızı alır ve istenildiğinde `/undo` ile tek adımda geri alır.

---

## 🛠️ Adım Adım Kurulum Kılavuzu

Kurulum sırasında herhangi bir hata almamak için aşağıdaki adımları sırasıyla uygulayın.

### 1. Ön Gereksinimler
* **Python 3.10 veya üzeri** (Python 3.11+ önerilir).
  * Kontrol etmek için terminalde `python --version` veya Windows'ta `py -3 --version` komutunu çalıştırın.
* **Git** (projeyi klonlamak ve sürüm kontrol araçları için).
* **EVREN AI API Anahtarı** (`evren_llm_` ile başlayan kişisel API anahtarınız).

---

### 2. Projeyi İndirin ve Kurun

Terminalinizi (PowerShell, CMD, Bash veya Zsh) açın ve projeyi klonlayıp dizine girin:

```bash
git clone https://github.com/sezerdemir7/evren-code.git
cd evren-code
```

#### Seçenek A: Doğrudan Kurulum (Hızlı)
**Windows (PowerShell / CMD):**
```powershell
py -3 -m pip install --upgrade pip
py -3 -m pip install -e .
```

**Linux / macOS:**
```bash
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```

#### Seçenek B: Sanal Ortam (Virtual Environment) ile İzole Kurulum (Önerilen)
Sisteminizdeki diğer Python paketleriyle çakışma yaşanmaması için sanal ortam kullanabilirsiniz:

**Windows (PowerShell):**
```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

---

### 3. EVREN API Anahtarı Alma ve Tanımlama (İlk Kurulum)

#### Adım 3.1: EVREN Platformundan API Anahtarı Oluşturma
1. Tarayıcınızdan **[https://evren.ssyz.org.tr/](https://evren.ssyz.org.tr/)** adresine gidin ve hesabınızla giriş yapın.
2. Sol menüden veya üst panelden **LLM Çıkarım** bölümüne girin.
3. **API Anahtarı** sekmesine tıklayın ve **API Anahtarı Oluştur** butonuna basın.
4. Sistemin sizin için ürettiği `evren_llm_...` ile başlayan anahtarı kopyalayın.

#### Adım 3.2: API Anahtarını Araca Tanımlama
Aldığınız `evren_llm_...` anahtarını iki farklı yöntemle tanımlayabilirsiniz:

##### Yöntem 1: İnteraktif Kurulum Sihirbazı (Önerilen)
Terminalde aşağıdaki komutu çalıştırın:
```bash
evren --setup
```
*(Veya doğrudan `evren` yazıp çalıştırın; kayıtlı anahtarınız yoksa sihirbaz otomatik açılır.)*

Kurulum sihirbazı sırasıyla şunları yapar:
1. `evren` komutunun her dizinden çalışabilmesi için Python `Scripts`/`bin` klasörünü işletim sistemi `PATH` değişkeninize otomatik ekler.
2. `evren_llm_...` API anahtarınızı ister (yapıştırırken güvenlik nedeniyle ekranda karakterler görünmez, yapıştırıp `Enter`'a basmanız yeterlidir) ve formatını doğrular.
3. İlk kez kullanıyorsanız **EVREN Kullanım Şartları** metnini ekrana getirir ve onayınızla (`e`) API üzerinden sözleşme onayını (`/v1/terms/accept`) otomatik tamamlar.
4. Aktif modelleri keşfeder ve anahtarınızı `~/.evren-code/.evren-key` dosyasına güvenli şekilde kaydeder.

##### Yöntem 2: `.env` Dosyası veya Ortam Değişkeni ile Tanımlama
İnteraktif sihirbaz yerine `.env` dosyası kullanmak isterseniz:
```bash
# Örnek dosyayı kopyalayın
cp .env.example .env
```
`.env` dosyasını açıp anahtarınızı yapıştırın:
```env
EVREN_API_KEY=evren_llm_sizin_anahtariniz
EVREN_BASE_URL=https://evren-llmapi.ssyz.org.tr/v1
EVREN_DEFAULT_MODEL=deepseek-v4.1-flash
```

---

### 4. API Anahtarının Süresi Bittiğinde / Yenilenmesi Gerektiğinde Nasıl Güncellenir?

EVREN platformundaki API anahtarınızın kullanım süresi dolduğunda veya anahtarınızı iptal edip yenilediğinizde araç `401 Unauthorized` (Kimlik Doğrulama Hatası) uyarısı verecektir. Yeni anahtarınızı tanımlamak için:

1. **[https://evren.ssyz.org.tr/](https://evren.ssyz.org.tr/)** adresine girin -> **LLM Çıkarım** -> **API Anahtarı** bölümünden yeni bir `evren_llm_...` anahtarı oluşturup kopyalayın.
2. **Yeni anahtarı kaydetmek için aşağıdaki yollardan birini uygulayın:**
   * **Seçenek A — İnteraktif Oturum İçinden Çıkmadan (`/auth`):**
     Açık bir `evren` oturumu sırasındayken konuşma geçmişinizi kaybetmeden doğrudan şu komutu yazın:
     ```text
     /auth
     ```
     *(Alternatif olarak `/key` takma adını da kullanabilir, mevcut anahtarın maskelenmiş durumunu görmek için `/auth status` yazabilirsiniz.)* Yeni `evren_llm_...` anahtarınızı gizli girdi ekranına yapıştırıp `Enter`'a bastığınızda bağlantı doğrulanır, `~/.evren-code/.evren-key` güncellenir ve oturum kesintisiz devam eder.
   * **Seçenek B — Terminalden Tek Komutla (`evren --setup`):**
     ```bash
     evren --setup
     ```
     komutunu çalıştırın ve yeni `evren_llm_...` anahtarınızı yapıştırıp `Enter`'a basın. Eski anahtarın üzerine otomatik olarak yazılacak ve bağlantı anında doğrulanacaktır.
   * **Seçenek C — `.env` Dosyası Kullanıyorsanız:**
     Proje dizinindeki `.env` dosyasını açıp `EVREN_API_KEY=evren_llm_yeni_anahtariniz` satırını güncelleyin. *(Not: Ortam değişkeninde veya `.env` içinde eski bir `EVREN_API_KEY` tanımlıysa öncelikli olarak o okunur; bu yüzden `.env` kullanıyorsanız oradaki değeri de güncellemeyi unutmayın.)*
   * **Seçenek D — Anahtar Dosyasını Doğrudan Düzenleme:**
     Kullanıcı ana dizininizdeki `~/.evren-code/.evren-key` (Windows'ta `C:\Users\<KullaniciAdiniz>\.evren-code\.evren-key`) dosyasını bir metin editörüyle açıp içine doğrudan yeni `evren_llm_...` anahtarınızı yapıştırıp kaydedebilirsiniz.

---

### 5. Yeni Güncellemeleri Alma (Evren-Code'u Güncelleme)

GitHub deposuna yeni özellikler, model güncellemeleri veya hata düzeltmeleri eklendiğinde aracınızı en güncel sürüme yükseltmek için iki yöntem kullanabilirsiniz:

#### Yöntem 1: Tek Komutla Otomatik Güncelleme (Önerilen)
Hangi klasörde olursanız olun terminalde şu komutu çalıştırın:
```bash
evren --update
```
Bu komut arka planda `evren-code` kaynak dizinine giderek GitHub'dan en son değişiklikleri (`git pull`) otomatik çeker ve yeni bağımlılıklar varsa (`pip install -e .`) günceller.

#### Yöntem 2: Git ile Manuel Güncelleme
Projeyi klonladığınız `evren-code` klasörüne giderek:
```bash
cd evren-code
git pull origin main
py -3 -m pip install -e .
```
*(Not: İlk kurulumu `-e` parametresiyle yaptığınız için `git pull` sonrasında yapılan kod değişiklikleri anında aktif olur.)*

---

### 6. Sık Karşılaşılan Kurulum Sorunları ve Çözümleri

* **Sorun 1: `'evren' terimi bir cmdlet, işlev veya program olarak tanınmıyor` (`command not found: evren`)**
  * **Nedeni:** Python'un `Scripts` klasörü henüz terminalinizin `PATH` ortam değişkeninde aktif olmamıştır.
  * **Çözümü:**
    1. Önce `py -3 -m evren_code.cli --setup` (veya `python -m evren_code.cli --setup`) komutunu çalıştırın. Bu komut `Scripts` dizinini `PATH`'e otomatik ekler.
    2. Değişikliğin etkili olması için **mevcut terminal penceresini kapatıp yeni bir terminal açın**.
    3. Alternatif olarak proje içindeki `evren.bat` dosyasını veya `py -3 -m evren_code.cli` komutunu her zaman doğrudan kullanabilirsiniz.

* **Sorun 2: `403 terms_not_accepted` (Kullanım Şartları Hatası)**
  * **Çözümü:** `evren --setup` komutunu çalıştırıp ekrana gelen kullanım şartlarına `e` (Evet) yanıtını verin; araç `/v1/terms/accept` onayını otomatik olarak gerçekleştirecektir.

* **Sorun 3: `401 Unauthorized` / `Geçersiz API Anahtarı` Hatası**
  * **Çözümü:** Anahtarınızın süresi dolmuş olabilir. Yukarıdaki **4. Bölüm** adımlarını takip ederek `https://evren.ssyz.org.tr/` -> **LLM Çıkarım** -> **API Anahtarı** ekranından yeni anahtar alıp açık oturum içinde `/auth` yazarak veya terminalde `evren --setup` çalıştırarak güncelleyin.

---

## 💻 Kullanım Rehberi

### 1. İnteraktif Kodlama Modu (REPL)
Üzerinde çalıştığınız herhangi bir yazılım projesinin klasörüne gidin ve `evren` komutunu çalıştırın:

```bash
cd /projemin/dizini
evren
```

İnteraktif ekranda bilmeniz gereken pratik özellikler:
* **Doğal Dille Görev Verme:** Özel bir komut ezberlemenize gerek yoktur. Doğrudan Türkçe olarak ne yapılmasını istediğinizi yazın:
  * *"Bu projenin mimarisini analiz et ve ana modülleri özetle"*
  * *"src/auth.py içindeki token doğrulama hatasını bul ve düzelt"*
  * *"Son yaptığımız değişiklikler için birim testlerini yaz ve çalıştır"*
* **`@dosya` ve `@clipboard` (Pano) Etiketleme:**
  * Mesaj içinde `@src/main.py` şeklinde dosya yolu belirterek o dosyanın içeriğini doğrudan bağlama ekleyebilirsiniz.
  * Ekran görüntüsü kopyaladığınızda dosyaya kaydetmeye gerek kalmadan `"Bu arayüz hatasını çöz @clipboard"` (veya `@pano`) yazarak panodaki görseli doğrudan `qwen3-vl-30b` Vision modeline gönderebilirsiniz (panoda metin varsa metni ekler).
* **Çoklu Satır Yapıştırma (Bracketed Paste):** Terminale çok satırlı uzun hata logları (stack trace) veya kod blokları yapıştırdığınızda ilk satırda bölünmeden tek blok halinde alınır (`Alt+Enter` ile alt satıra geçebilirsiniz).
* **`!komut` ile Hızlı Terminal Komutu:** Ajanı devreye sokmadan terminalde hızlıca komut çalıştırmak için başına `!` koyun (Örn: `!git status` veya `!pytest -q`).
* **Tur Sonu Mühendislik Özeti & Commit Asistanı:** Her tur sonunda düzenlenen dosya sayısı, eklenen/silinen satır (`+X / -Y`), geçen testler, süre ve token özeti otomatik gösterilir; ayrıca `git_commit_assistant` aracıyla yapılan değişiklikler için Conventional Commit ve PR özeti taslağı oluşturulabilir.
* **Değişiklik Onay Kapısı (`[E/h/t]`):** Ajan bir dosyayı düzenlemeden veya kabuk komutu çalıştırmadan önce size Diff önizlemesini gösterir:
  * `Enter` veya `e`: İşlemi onayla
  * `h`: İşlemi reddet
  * `t`: Bu oturum boyunca tüm işlemlere güven (tekrar onay sorma)

---

### 2. İnteraktif Slash (`/`) Komutları

İnteraktif oturum sırasında `/` tuşuna bastığınızda otomatik tamamlama menüsü açılır:

| Komut | Açıklama |
|:---|:---|
| `/model [id\|rol]` | Aktif modeli görüntüler veya değiştirir (Örn: `/model coder`, `/model architect`, `/model glm-5.3`) |
| `/auth [status]` | Oturumdan çıkmadan EVREN API anahtarını gizli girdi ile günceller veya maskelenmiş durumunu gösterir (`/key` takma adı da desteklenir) |
| `/undo` | Ajanın yaptığı son dosya değişikliğini anında geri alır (Snapshot Rollback) |
| `/status` | Aktif modeli, mesaj sayısını, tahmini token kullanımını ve geri alma (`/undo`) sayısını gösterir |
| `/compact` | Konuşma geçmişini özetleyerek bağlam penceresinde (context) yer açar |
| `/clear` | Konuşma geçmişini, aktif planı ve oturum belleğini sıfırlar |
| `/help` | Temel komut kılavuzunu ekrana basar |
| `/exit` | Oturumu çalışma alanına kaydeder ve güvenli şekilde çıkar |

---

### 3. Tek Seferlik Komut Satırı Kullanımı (One-Shot)

İnteraktif moda girmeden doğrudan tek bir görevi çalıştırıp çıkmak için:

```bash
# Geçerli dizinde görev çalıştırma
evren "Projedeki tüm Python dosyalarını tara ve eksik docstring'leri tamamla"

# Belirli bir proje dizininde (-C) ve onay sormadan (--trust) çalıştırma
evren -C "C:/projelerim/backend-app" --trust "Testleri çalıştır ve başarısız olanları düzelt"

# Belirli bir model (-m) veya uzmanlık rolü (--role) seçerek çalıştırma
evren --role architect "Veritabanı şeması ve önbellekleme mimarisini incele"
evren -m qwen3.8-flash-next "utils.py içine hızlı bir LRU cache sınıfı ekle"
```

---

### 4. Doğrudan Saf LLM Sorgu Modu (`evren ask`)

Proje dosyalarına dokunmadan, ajan araç döngüsüne girmeden doğrudan EVREN modellerine soru sormak veya kod parçacığı üretmek için `evren ask` (veya `--llm`) modunu kullanabilirsiniz:

```bash
# DeepSeek V4.1 Flash ile derin muhakeme / algoritma sorusu
evren ask -m deepseek-v4.1-flash "PostgreSQL'de B-Tree ve GIN indeksleri ne zaman tercih edilmeli?"

# Qwen 3.8 Flash Next ile hızlı kod üretimi (JSON çıktılı)
evren ask --json -m qwen3.8-flash-next "Python'da exponential backoff destekli async retry decorator yaz"
```

---

### 5. Otomasyon, CI/CD ve Alt-Ajan (Headless JSON) Modu

Claude Code, Antigravity, Codex veya kendi otomasyon betikleriniz içinden `evren-code`'u onay beklemeyen bir alt-ajan olarak çağırmak için `--headless --json` parametrelerini kullanın:

```bash
evren --headless --json --role coder -C "/hedef/proje" "JWT refresh token rotasyonunu ekle"
```

`stdout` üzerinden makine tarafından okunabilir standart JSON yanıtı döner:
```json
{
  "status": "success",
  "task": "JWT refresh token rotasyonunu ekle",
  "model_used": "qwen3.8-flash-next",
  "role": "coder",
  "workspace": "/hedef/proje",
  "summary": "Refresh token rotasyonu ve birim testleri başarıyla eklendi.",
  "modified_files": [
    "src/auth.py",
    "tests/test_auth.py"
  ]
}
```

---

### 6. Model Context Protocol (MCP) Sunucusu & Harici MCP İstemcisi (Client)

#### A. Evren-Code'u MCP Sunucusu Olarak Kullanma (`evren mcp`)
**Claude Desktop**, **Claude Code**, **Antigravity** veya **Cursor** gibi araçlara EVREN AI modellerini yerel bir MCP sunucusu olarak bağlamak için:

```bash
evren mcp
```

* **Sunulan MCP Araçları:**
  * `evren_llm_call(prompt, model, system_prompt, include_thinking)`: Dosya değişikliği yapmadan doğrudan saf LLM API çağrısı.
  * `evren_task(task, workspace, role)`: Otonom kodlama, hata ayıklama ve refactoring görevi.
  * `evren_architect(task, workspace)`: DeepSeek V4.1 Flash ile mimari tasarım ve risk analizi.
  * `evren_code_review(path, workspace)`: Güvenlik ve kalite odaklı kod incelemesi.
  * `evren_models()`: Aktif model kataloğunu, rolleri ve bağlam limitlerini listeleme.
* **Hazır Entegrasyon Şablonları:** `integrations/` klasörü altında Claude Desktop (`claude_desktop_config.json`), Claude Code (`CLAUDE.md`), Antigravity (`antigravity_skill.md`) ve Python/Codex (`codex_agent_runner.py`) için hazır şablonlar yer almaktadır.

#### B. Harici MCP Sunucularına Bağlanma (MCP Client — `.evren/mcp.json`)
Projenizdeki `.evren/mcp.json` veya global `~/.evren-code/mcp.json` dosyasına harici MCP sunucuları (PostgreSQL, GitHub, SQLite, Playwright vb.) tanımladığınızda Evren-Code başlangıçta bu sunuculara otomatik bağlanır ve araçlarını `mcp_<sunucu>_<arac>` adıyla kendi araç havuzuna ekler:

```json
{
  "mcpServers": {
    "sqlite": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sqlite", "app.db"]
    }
  }
}
```

---

## 📊 Geliştirme Günlükleri (Trajectory Logging)

Ajanın çalışma sırasında attığı adımlar, çalıştırdığı araçlar ve düşünme (`<think>`) süreçleri projenin `.evren/logs/` dizininde (`trajectory.jsonl` ve `session_journal.md`) saklanır:
* Projede `.gitignore` dosyası varsa `.evren/` dizini otomatik olarak `.gitignore` içine eklenir; git deponuz kirlenmez.
* Günlük tutulmasını istemiyorsanız `evren --no-log` bayrağı ile başlatabilirsiniz.

---

## 🧪 Test ve Doğrulama

Birim ve entegrasyon testlerini çalıştırmak için:

```bash
py -3 -m pytest -q
```

Canlı API bağlantısı gerektirmeyen deterministik değerlendirme (eval) benchmark testini çalıştırmak için:

```bash
py -3 evals/runner.py --mock
```

---

## 📄 Lisans

Bu proje MIT Lisansı altında açık kaynak olarak paylaşılmıştır.
