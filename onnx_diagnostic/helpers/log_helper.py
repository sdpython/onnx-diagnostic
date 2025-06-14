import re
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import pandas


class CubeLogs:
    """
    Processes logs coming from experiments.
    """

    def __init__(
        self,
        data: Any,
        time: str = "date",
        keys: Sequence[str] = ("version_.*", "model_.*"),
        values: Sequence[str] = ("time_.*", "disc_.*"),
        ignored: Sequence[str] = (),
        recent: bool = False,
        formulas: Optional[Dict[str, Callable[[pandas.DataFrame], pandas.Series]]] = None,
    ):
        self._data = data
        self._time = time
        self._keys = keys
        self._values = values
        self._ignored = ignored
        self.recent = recent
        self._formulas = formulas

    def load(self, verbose: int = 0):
        """Loads and preprocesses the data."""
        if isinstance(self._data, pandas.DataFrame):
            if verbose:
                print(f"[CubeLogs.load] load from dataframe, shape={self._data.shape}")
            self.data = self._data
        elif isinstance(self._data, list) and all(isinstance(r, dict) for r in self._data):
            if verbose:
                print(f"[CubeLogs.load] load from list of dicts, n={len(self._data)}")
            self.data = pandas.DataFrame(self._data)
        else:
            raise NotImplementedError(
                f"Not implemented with the provided data (type={type(self._data)})"
            )

        assert all(isinstance(c, str) for c in self.data.columns), (
            f"The class only supports string as column names "
            f"but found {[c for c in self.data.columns if not isinstance(c, str)]}"
        )
        if verbose:
            print(f"[CubeLogs.load] loaded with shape={self.data.shape}")

        self._initialize_columns()
        if verbose:
            print(f"[CubeLogs.load] time={self.time}")
            print(f"[CubeLogs.load] keys={self.keys}")
            print(f"[CubeLogs.load] values={self.values}")
            print(f"[CubeLogs.load] ignored={self.ignored}")
            print(f"[CubeLogs.load] ignored_values={self.ignored_values}")
            print(f"[CubeLogs.load] ignored_keys={self.ignored_keys}")
        assert not (
            set(self.keys) & set(self.values)
        ), f"Columns {set(self.keys) & set(self.values)} cannot be keys and values"
        assert not (
            set(self.keys) & set(self.ignored)
        ), f"Columns {set(self.keys) & set(self.ignored)} cannot be keys and ignored"
        assert not (
            set(self.values) & set(self.ignored)
        ), f"Columns {set(self.keys) & set(self.ignored)} cannot be values and ignored"
        assert (
            self.time not in self.keys
            and self.time not in self.values
            and self.time not in self.ignored
        ), f"Column {self.time!r} is also a key, a value or ignored"
        self._columns = [self.time, *self.keys, *self.values, *self.ignored]
        self.dropped = [c for c in self.data.columns if c not in set(self.columns)]
        self.data = self.data[self.columns]
        if verbose:
            print(f"[CubeLogs.load] dropped={self.dropped}")
            print(f"[CubeLogs.load] data.shape={self.data.shape}")

        self._preprocess()
        if self.recent and verbose:
            print(f"[CubeLogs.load] keep most recent data.shape={self.data.shape}")

        # Let's apply the formulas
        if self._formulas:
            for k, f in self._formulas.items():
                if verbose:
                    print(f"[CubeLogs.load] apply formaula {k!r}")
                self.data[k] = f(self.data)
        self.values_for_key = {k: set(self.data[k]) for k in self.keys}
        nans = [c for c in self.keys if self.data[c].isna().astype(int).sum() > 0]
        assert not nans, f"The following keys {nans} have nan values. This is not allowed."
        if verbose:
            print(f"[CubeLogs.load] done, shape={self.shape}")

    @property
    def shape(self) -> Tuple[int, int]:
        "Returns the shape."
        assert hasattr(self, "data"), "Method load was not called"
        return self.data.shape

    @property
    def columns(self) -> Sequence[str]:
        "Returns the columns."
        assert hasattr(self, "data"), "Method load was not called"
        return self.data.columns

    def _preprocess(self):
        last = self.values[0]
        gr = self.data[[self.time, *self.keys, last]].groupby([self.time, *self.keys]).count()
        gr = gr[gr[last] > 1]
        assert gr.shape[0] == 0, f"There are duplicated rows:\n{gr}"
        if self.recent:
            gr = self.data[[*self.keys, self.time]].groupby(self.keys, as_index=False).max()
            filtered = pandas.merge(self.data, gr, on=[self.time, *self.keys])
            assert filtered.shape[0] <= self.data.shape[0], (
                f"Keeping the latest row brings more row {filtered.shape} "
                f"(initial is {self.data.shape})."
            )
            self.data = filtered
        else:
            gr = self.data[[*self.keys, self.time]].groupby(self.keys).count()
            gr = gr[gr[self.time] > 1]
            assert (
                gr.shape[0] == 0
            ), f"recent should be true to keep the most recent row:\n{gr}"

    @classmethod
    def _filter_column(cls, filters, columns, can_be_empty=False):
        set_cols = set()
        for f in filters:
            reg = re.compile(f)
            cols = [c for c in columns if reg.search(c)]
            set_cols |= set(cols)
        assert (
            can_be_empty or set_cols
        ), f"Filters {filters} returns an empty set from {columns}"
        return sorted(set_cols)

    def _initialize_columns(self):
        self.keys = self._filter_column(self._keys, self.data.columns)
        self.values = self._filter_column(self._values, self.data.columns)
        self.ignored = self._filter_column(self._ignored, self.data.columns, True)
        assert (
            self._time in self.data.columns
        ), f"Column {self._time} not found in {self.data.columns}"
        ignored_keys = set(self.ignored) & set(self.keys)
        ignored_values = set(self.ignored) & set(self.values)
        self.keys = [c for c in self.keys if c not in ignored_keys]
        self.values = [c for c in self.values if c not in ignored_values]
        self.ignored_keys = sorted(ignored_keys)
        self.ignored_values = sorted(ignored_values)
        self.time = self._time

    def __str__(self) -> str:
        "usual"
        return str(self.data) if hasattr(self, "data") else str(self._data)

    def view(
        self,
        key_index: Sequence[str],
        values: Sequence[str],
        ignore_unique: bool = True,
        order: Optional[Sequence[str]] = None,
    ) -> pandas.DataFrame:
        """
        Returns a dataframe, a pivot view.
        `key_index` determines the index, the other key columns determines
        the columns. If `ignore_unique` is True, every columns with a unique value
        is removed.

        :param key_index: keys to put in the row index
        :param values: values to show
        :param ignore_unique: ignore keys with a unique value
        :param order: to reorder key in columns index
        :return: dataframe
        """
        key_index = self._filter_column(key_index, self.keys)
        values = self._filter_column(values, self.values)
        assert set(key_index) <= set(
            self.keys
        ), f"Non existing columns in key_index {set(key_index) - set(self.keys)}"
        assert set(values) <= set(
            self.values
        ), f"Non existing columns in values {set(values) - set(self.values)}"
        set_key_columns = {c for c in self.keys if c not in key_index}
        if ignore_unique:
            key_index = [k for k in key_index if len(self.values_for_key[k]) > 1]
            key_columns = [k for k in set_key_columns if len(self.values_for_key[k]) > 1]
        else:
            key_columns = sorted(set_key_columns)
        if order:
            assert set(order) <= set_key_columns, (
                f"Non existing columns from order in key_columns "
                f"{set(order) - set_key_columns}"
            )
            key_columns = [*order, *[c for c in key_columns if c not in order]]
        return self.data.pivot(index=key_index[::-1], columns=key_columns, values=values)
