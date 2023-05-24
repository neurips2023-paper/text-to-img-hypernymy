import sys
import pathlib
import json
import tqdm
import numpy as np
import pandas as pd


class LemmaCounter:
    def __init__(self, synset_lemmas_path):
        with open(synset_lemmas_path) as synset_lemmas_file:
            self.synset_lemmas = json.load(synset_lemmas_file)

        self.all_lemmas = [
            lemma.lower() for lemma in
            sum(self.synset_lemmas.values(), start=[])
        ]

    def count_lemmas(self, parquet_path: pathlib.Path, save_path: str, part_id: str):
        all_texts = pd.read_parquet(parquet_path).TEXT.str.lower()

        lemma_counts = np.zeros(len(self.all_lemmas), dtype=np.int32)
        for text in tqdm.tqdm(all_texts):
            lemma_counts += np.asarray(
                [lemma in (text if text is not None else "") for lemma in self.all_lemmas],
                dtype=np.int32,
            )

        save_path = pathlib.Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        np.save(save_path / f"{part_id}.npy", lemma_counts)


def main():
    print(sys.argv)
    _, synset_lemmas_path, parquet_path, save_path, part_id = sys.argv
    lemma_counter = LemmaCounter(synset_lemmas_path)
    lemma_counter.count_lemmas(parquet_path, save_path, part_id)


if __name__ == "__main__":
    main()

