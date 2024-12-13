from haystack_integrations.components.rankers.fastembed import FastembedRanker
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.generators import OpenAIGenerator
from haystack.components.converters import HTMLToDocument
from haystack.components.builders import PromptBuilder
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret
from haystack import Pipeline
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_TOKEN = os.getenv("OPENAI_TOKEN")

template = """Given the information below, answer the query. Only use the provided context to generate the answer and output the used document links
Context:
{% for document in documents %}
{{ document.content }}
URL: {{ document.meta.url }}
{% endfor %}

Question: {{ query }}
Answer:"""

splitter = DocumentSplitter(split_by="word", split_length=10, split_overlap=3)
document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
prompt_builder = PromptBuilder(template=template)
writer = DocumentWriter(document_store=document_store)
generator = OpenAIGenerator(
    model="gpt-4o-mini", api_key=Secret.from_token(OPENAI_TOKEN)
)
fetcher = LinkContentFetcher()
converter = HTMLToDocument()
ranker = FastembedRanker()


indexing_pipeline = Pipeline()
indexing_pipeline.add_component(instance=prompt_builder, name="prompt_builder")
indexing_pipeline.add_component(instance=converter, name="converter")
indexing_pipeline.add_component(instance=generator, name="generator")
indexing_pipeline.add_component(instance=splitter, name="splitter")
indexing_pipeline.add_component(instance=fetcher, name="fetcher")
indexing_pipeline.add_component(instance=ranker, name="ranker")

indexing_pipeline.connect("fetcher.streams", "converter.sources")
indexing_pipeline.connect("converter.documents", "splitter.documents")
indexing_pipeline.connect("splitter.documents", "ranker.documents")
indexing_pipeline.connect("ranker.documents", "prompt_builder.documents")
indexing_pipeline.connect("prompt_builder", "generator")

query = "Explain about the person of Alif in funny way?"
result = indexing_pipeline.run(
    data={
        "fetcher": {"urls": ["https://alif.top"]},
        "prompt_builder": {
            "query": query,
        },
        "ranker": {"query": query, "top_k": 4},
    }
)

print(result)
