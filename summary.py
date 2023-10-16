from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import WikipediaLoader
from langchain.chat_models.gigachat import GigaChat

from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import load_prompt
from langchain.document_loaders import PyPDFLoader

# Загрузка из википедии
# docs = WikipediaLoader(query="Московское центральное кольцо", lang="ru", load_max_docs=1, doc_content_chars_max=1000000).load()

# Загрузка из pdf
docs = PyPDFLoader("Московское_центральное_кольцо.pdf").load()

# Загружаем промпты из хаба. Можно сохранить их локально и загружать из файлов, можно написать свои
map_prompt = load_prompt('lc://prompts/summarize/map_reduce/map.yaml')
combine_prompt = load_prompt('lc://prompts/summarize/map_reduce/combine.yaml')

# Размеры чанков и перекрытие выбираются исходя из возможностей модели
split_docs = CharacterTextSplitter(chunk_size=15000, chunk_overlap=1000).split_documents(docs)
# Инициализируем гигачат
giga = GigaChat(
    # Здесь или в env прописываем креды
    profanity=False, # Отключение цензора
    verbose=True, # Логи запросов
    base_url="https://wmapi-dev.saluteai-pd.sberdevices.ru/v1", # Если не указывать, будет пром
    model="GigaChat-70b-4k-base", # моделька
    verify_ssl_certs=False) # Для работы без серта минцифры

if len(split_docs) == 1:
    # Если вся дата влезла в один чанк - суммаризируем одним запросом
    chain = load_summarize_chain(giga, chain_type="stuff", prompt=combine_prompt)
else:
    # Если дата разбилась на несколько чанков - суммаризируем через map-reduce
    chain = load_summarize_chain(
        giga,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
    )


res = chain.run(split_docs)

print(res)

# Результат (для википедии):
# 1. Малое кольцо Московской железной дороги было построено в XIX веке и использовалось для перевозки грузов и пассажиров.
# 2. В 2016 году Малое кольцо было полностью реконструировано и электрифицировано, и на его основе был запущен новый вид общественного транспорта — Московское центральное кольцо (МЦК).
# 3. МЦК интегрировано с метрополитеном, что позволяет пассажирам пересаживаться между ними без дополнительной оплаты.
# 4. МЦК имеет 31 остановочный пункт с высокими платформами, которые являются лишь остановочными пунктами (платформами) и не являются железнодорожными станциями.
# 5. Основным видом подвижного состава на МЦК являются электропоезда ЭС2Г «Ласточка», которые имеют салоны пригородно-городского исполнения на 386 или 346 сидений.
# 6. МЦК соединяет несколько районов Москвы и обеспечивает удобную пересадку на другие виды общественного транспорта.
# 7. МЦК является популярным видом транспорта среди пассажиров благодаря своей удобной локации и высокому уровню комфорта.