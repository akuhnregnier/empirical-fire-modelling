# -*- coding: utf-8 -*-
"""Custom Joblib backend."""
# -*- coding: utf-8 -*-
import ast
import datetime
import logging
import os
import re
import shutil
import traceback

import cloudpickle
from joblib import register_store_backend
from joblib._store_backends import (
    CacheItemInfo,
    StoreBackendBase,
    StoreBackendMixin,
    concurrency_safe_rename,
    concurrency_safe_write,
    mkdirp,
    rm_subdirs,
)
from lazy_object_proxy.slots import Proxy
from wildfires.data import get_hash

logger = logging.getLogger(__name__)


def custom_get_hash(arg):
    """Compute the hash with support for lazy proxied objects."""
    if isinstance(arg, HashProxy):
        return arg.hashed_value
    return get_hash(arg)


class Factory:
    """Factory function that keeps track of whether it was called."""

    __slots__ = ("factory", "_was_called")

    def __init__(self, factory):
        self.factory = factory
        self._was_called = False

    @property
    def was_called(self):
        return self._was_called

    def __call__(self, *args, **kwargs):
        self._was_called = True
        return self.factory(*args, **kwargs)


class HashProxy(Proxy):
    """Lazy proxy containing a pre-calculated hash value."""

    __slots__ = "_hash_value"

    def __init__(self, factory, hash_value=None):
        """Initialise the proxy with the factory function and pre-defined hash.

        Args:
            factory (Factory): Factory function that generates the object to be
                proxied on demand.
            hash_value (int): Pre-computed hash value. This will be invalidated once
                `factory` is called. If None is given, the hash value will be computed
                every time (which required calling `factory`).

        """
        if not isinstance(factory, Factory):
            raise TypeError("'factory' needs to be a Factory instance.")
        super().__init__(factory)
        self._hash_value = hash_value

    @property
    def hashed_value(self):
        if self._hash_value is None or self.__factory__.was_called:
            self._hash_value = None  # Ensure this will never be accessed.
            return get_hash(self.__wrapped__)
        return self._hash_value


def register_backend():
    """Register custom backend for use with joblib memory."""
    register_store_backend("custom", CustomStoreBackend)


class DeletedError(Exception):
    pass


class CustomStoreBackend(StoreBackendBase, StoreBackendMixin):
    """A StoreBackend used with local or network file systems."""

    _open_item = staticmethod(open)
    _item_exists = staticmethod(os.path.exists)
    _move_item = staticmethod(concurrency_safe_rename)

    def clear_location(self, location):
        """Delete location on store."""
        if location == self.location:
            rm_subdirs(location)
        else:
            shutil.rmtree(location, ignore_errors=True)

    def create_location(self, location):
        """Create object location on store"""
        mkdirp(location)

    def get_items(self):
        """Returns the whole list of items available in the store."""
        items = []

        for dirpath, _, filenames in os.walk(self.location):
            is_cache_hash_dir = re.match("[a-f0-9]{32}", os.path.basename(dirpath))

            if is_cache_hash_dir:
                try:
                    for filename in ("output.cpkl", "hash_values.txt"):
                        output_filename = os.path.join(dirpath, filename)
                        try:
                            last_access = os.path.getatime(output_filename)
                        except OSError:
                            try:
                                last_access = os.path.getatime(dirpath)
                            except OSError:
                                # The directory has already been deleted
                                raise DeletedError()
                except DeletedError:
                    # The directory has already been deleted
                    continue

                last_access = datetime.datetime.fromtimestamp(last_access)
                try:
                    full_filenames = [os.path.join(dirpath, fn) for fn in filenames]
                    dirsize = sum(os.path.getsize(fn) for fn in full_filenames)
                except OSError:
                    # Either output_filename or one of the files in
                    # dirpath does not exist any more. We assume this
                    # directory is being cleaned by another process already
                    continue

                items.append(CacheItemInfo(dirpath, dirsize, last_access))

        return items

    def configure(self, location, verbose=1, backend_options=None):
        """Configure the store backend."""
        if backend_options is None:
            backend_options = {}

        # setup location directory
        self.location = location
        if not os.path.exists(self.location):
            mkdirp(self.location)

        self.verbose = verbose

    def load_item(self, path, verbose=1, msg=None):
        """Load an item from the store given its path as a list of strings."""
        full_path = os.path.join(self.location, *path)

        if verbose > 1:
            if verbose < 10:
                print("{0}...".format(msg))
            else:
                print("{0} from {1}".format(msg, full_path))

        filename = os.path.join(full_path, "output.cpkl")
        if not self._item_exists(filename):
            raise KeyError(
                "Non-existing item (may have been "
                "cleared).\nFile %s does not exist" % filename
            )

        hash_filename = os.path.join(full_path, "hash_values.txt")
        if not self._item_exists(hash_filename):
            raise KeyError(
                "Non-existing item (may have been "
                "cleared).\nFile %s does not exist" % hash_filename
            )

        with open(hash_filename, "r") as f:
            hash_values = ast.literal_eval(f.read())

        logger.debug(f"Creating Factory from filename '{filename}'.")
        logger.debug(f"Loaded hash values: {hash_values}.")

        def load_func():
            if hasattr(load_func, "stored"):
                logger.debug("Returning stored data.")
                return load_func.stored

            logger.debug(f"cloudpickle loading filename '{filename}'.")
            with open(filename, "rb") as f:
                contents = cloudpickle.load(f)

            load_func.stored = contents
            return contents

        if len(hash_values) == 1:
            # If only a single value has been stored.
            return HashProxy(Factory(load_func), hash_value=hash_values[0])
        # Otherwise create a lazy proxy for each individual object to associate each
        # stored object with its individual hash value.

        def get_factory_func(i):
            def factory_func():
                return load_func()[i]

            return factory_func

        return tuple(
            HashProxy(Factory(get_factory_func(i)), hash_value=hash_value)
            for i, hash_value in enumerate(hash_values)
        )

    def dump_item(self, path, item, verbose=1):
        """Dump an item in the store at the path given as a list of
        strings."""
        try:
            item_path = os.path.join(self.location, *path)
            if not self._item_exists(item_path):
                self.create_location(item_path)
            filename = os.path.join(item_path, "output.cpkl")
            logger.debug(f"Caching '{item}' in '{item_path}'.")

            hash_filename = os.path.join(item_path, "hash_values.txt")
            logger.debug(f"Writing hash values to '{item_path}'.")

            if verbose > 10:
                print("Persisting in %s" % item_path)

            def write_func(to_write, dest_filename):
                with open(dest_filename, "wb") as f:
                    cloudpickle.dump(to_write, f, protocol=-1)

            self._concurrency_safe_write(item, filename, write_func)

            def hash_write_func(to_write, hash_filename):
                if not isinstance(to_write, tuple):
                    to_write = (to_write,)

                with open(hash_filename, "w") as f:
                    f.write(str(list(map(custom_get_hash, to_write))))

            self._concurrency_safe_write(item, hash_filename, hash_write_func)

        except:  # noqa: E722
            " Race condition in the creation of the directory "

    def _concurrency_safe_write(self, to_write, filename, write_func):
        """Writes an object into a file in a concurrency-safe way."""
        try:
            temporary_filename = concurrency_safe_write(to_write, filename, write_func)
        except:
            print("Something went wrong before moving the file.!")
            traceback.print_exc()
        self._move_item(temporary_filename, filename)

    def contains_item(self, path):
        """Check if there is an item at the path, given as a list of strings"""
        item_path = os.path.join(self.location, *path)
        filename = os.path.join(item_path, "output.cpkl")
        hash_filename = os.path.join(item_path, "hash_values.txt")

        return self._item_exists(filename) and self._item_exists(hash_filename)
