from sentence_transformers import SentenceTransformer
import numpy as np
from cfg import EMBEDDING_MODEL, DEVICE, EMBED_BATCH_SIZE
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"



class Embedder:
    def __init__(self, model_name=EMBEDDING_MODEL, device=DEVICE):
        self.model = SentenceTransformer(model_name, device=device)

    def embed_texts(self, texts, batch_size=EMBED_BATCH_SIZE):
        embeddings = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
        return embeddings  


class EmbeddingVisualizer:
    def __init__(
        self,
        pca_components=50,
        tsne_components=2,
        perplexity=5,
        n_iter=1000,
        random_state=42
    ):
        self.pca_components = pca_components
        self.tsne_components = tsne_components
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.random_state = random_state

        self.pca = None
        self.tsne = None

    def fit_transform(self, embeddings):
        embeddings = np.array(embeddings)

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")

        n_samples, n_features = embeddings.shape

        n_pca_components = min(self.pca_components, n_samples-1, n_features)

        self.pca = PCA(
            n_components=n_pca_components,
            random_state=self.random_state
        )
        pca_result = self.pca.fit_transform(embeddings)

        if n_samples < 3:
            print(
                "Warning: Too few samples for t-SNE. "
                "Returning PCA result instead."
            )
            return pca_result

        self.tsne = TSNE(
            n_components=self.tsne_components,
            perplexity=min(self.perplexity, n_samples - 1),
            n_iter=self.n_iter,
            random_state=self.random_state,
            method="exact",
            learning_rate="auto"
        )
        tsne_result = self.tsne.fit_transform(pca_result)

        return tsne_result

    def plot(self, reduced_result, title="Embedding Visualization"):
        plt.figure(figsize=(8, 6))

        if reduced_result.shape[1] == 1:
            plt.scatter(reduced_result[:, 0], np.zeros(len(reduced_result)))
            plt.ylabel("Zero Axis")
        else:
            plt.scatter(reduced_result[:, 0], reduced_result[:, 1])
            plt.ylabel("Dimension 2")

        plt.title(title)
        plt.xlabel("Dimension 1")   
        plt.show()


if __name__=="__main__":

    embedder = Embedder()
    u=embedder.embed_texts(["This is a test.", "This is a simple test.","Completely different sentence.","Another example sentence for embedding."])
    visualizer = EmbeddingVisualizer()
    tsne_result = visualizer.fit_transform(u)
    visualizer.plot(tsne_result)