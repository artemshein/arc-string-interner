#![cfg_attr(all(feature = "bench", test), feature(test))]
#![doc(html_root_url = "https://docs.rs/crate/arc-string-interner/0.1.0")]
#![deny(missing_docs)]

//! Caches strings efficiently, with minimal memory footprint and associates them with unique symbols.
//! These symbols allow constant time comparisons and look-ups to the underlying interned strings.
//!
//! ### Example: Interning & Symbols
//!
//! ```
//! use arc_string_interner::StringInterner;
//!
//! let mut interner = StringInterner::default();
//! let sym0 = interner.get_or_intern("Elephant");
//! let sym1 = interner.get_or_intern("Tiger");
//! let sym2 = interner.get_or_intern("Horse");
//! let sym3 = interner.get_or_intern("Tiger");
//! assert_ne!(sym0, sym1);
//! assert_ne!(sym0, sym2);
//! assert_ne!(sym1, sym2);
//! assert_eq!(sym1, sym3); // same!
//! ```
//!
// ### Example: Creation by `FromIterator`
//
// ```
// # use arc_string_interner::DefaultStringInterner;
// let interner = vec!["Elephant", "Tiger", "Horse", "Tiger"]
// 	.into_iter()
// 	.collect::<DefaultStringInterner>();
// ```
//!
//! ### Example: Look-up
//!
//! ```
//! # use arc_string_interner::StringInterner;
//! let mut interner = StringInterner::default();
//! let sym = interner.get_or_intern("Banana");
//! assert_eq!(interner.resolve(sym).map(|s| (&*s).to_string()), Some("Banana".to_owned()));
//! ```
//!
// ### Example: Iteration
//
// ```
// # use arc_string_interner::DefaultStringInterner;
// let interner = vec!["Earth", "Water", "Fire", "Air"]
// 	.into_iter()
// 	.collect::<DefaultStringInterner>();
// for (sym, str) in interner {
// 	// iteration code here!
// }
// ```

#[cfg(all(feature = "bench", test))]
extern crate test;

#[cfg(test)]
mod tests;

#[cfg(all(feature = "bench", test))]
mod benches;

// #[cfg(feature = "serde_support")]
// mod serde_impl;

use std::iter::FromIterator;
use std::{
    collections::{hash_map::RandomState, HashMap},
    hash::{BuildHasher, Hash, Hasher},
    iter, marker,
    num::NonZeroU32,
    slice,
    sync::Arc,
    vec,
};
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard, RwLockUpgradableReadGuard};
use std::collections::hash_map::Entry;
use init_with::InitWith;
use array_init::array_init;

/// Types implementing this trait are able to act as symbols for string interners.
///
/// Symbols are returned by `StringInterner::get_or_intern` and allow look-ups of the
/// original string contents with `StringInterner::resolve`.
///
/// # Note
///
/// Optimal symbols allow for efficient comparisons and have a small memory footprint.
pub trait Symbol: Copy + Ord + Eq {
    /// Creates a symbol from a `usize`.
    ///
    /// # Note
    ///
    /// Implementations panic if the operation cannot succeed.
    fn from_usize(val: usize) -> Self;

    /// Returns the `usize` representation of `self`.
    fn to_usize(self) -> usize;
}

/// Symbol type used by the `DefaultStringInterner`.
///
/// # Note
///
/// This special symbol type has a memory footprint of 32 bits
/// and allows for certain space optimizations such as using it within an option: `Option<Sym>`
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Sym(NonZeroU32);

impl Symbol for Sym {
    /// Creates a `Sym` from the given `usize`.
    ///
    /// # Panics
    ///
    /// If the given `usize` is greater than `u32::MAX - 1`.
    fn from_usize(val: usize) -> Self {
        assert!(
            val < u32::MAX as usize,
            "Symbol value {} is too large and not supported by `string_interner::Sym` type",
            val
        );
        Sym(NonZeroU32::new((val + 1) as u32).unwrap_or_else(|| {
            unreachable!("Should never fail because `val + 1` is nonzero and `<= u32::MAX`")
        }))
    }

    fn to_usize(self) -> usize {
        (self.0.get() as usize) - 1
    }
}

impl Symbol for usize {
    fn from_usize(val: usize) -> Self {
        val
    }

    fn to_usize(self) -> usize {
        self
    }
}

/// Internal reference to `str` used only within the `StringInterner` itself
/// to encapsulate the unsafe behaviour of interior references.
#[derive(Debug, Copy, Clone, Eq)]
struct InternalStrRef(*const str);

impl InternalStrRef {
    /// Creates an InternalStrRef from a str.
    ///
    /// This just wraps the str internally.
    fn from_str(val: &str) -> Self {
        InternalStrRef(val as *const str)
    }

    /// Reinterprets this InternalStrRef as a str.
    ///
    /// This is "safe" as long as this InternalStrRef only
    /// refers to strs that outlive this instance or
    /// the instance that owns this InternalStrRef.
    /// This should hold true for `StringInterner`.
    ///
    /// Does not allocate memory!
    fn as_str(&self) -> &str {
        unsafe { &*self.0 }
    }
}

impl<T> From<T> for InternalStrRef
where
    T: AsRef<str>,
{
    fn from(val: T) -> Self {
        InternalStrRef::from_str(val.as_ref())
    }
}

impl Hash for InternalStrRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_str().hash(state)
    }
}

impl PartialEq for InternalStrRef {
    fn eq(&self, other: &InternalStrRef) -> bool {
        self.as_str() == other.as_str()
    }
}

/// `StringInterner` that uses `Sym` as its underlying symbol type.
pub type DefaultStringInterner = StringInterner<Sym, RandomState, 8>;

/// Caches strings efficiently, with minimal memory footprint and associates them with unique symbols.
/// These symbols allow constant time comparisons and look-ups to the underlying interned strings.
#[derive(Debug)]
pub struct StringInterner<S, H, const N: usize>
where
    S: Symbol,
    H: BuildHasher,
{
    map: [RwLock<HashMap<InternalStrRef, S, H>>; N],
    values: [RwLock<Vec<Arc<str>>>; N],
    hash_builder: H,
}

// impl<S, H> PartialEq for StringInterner<S, H>
// where
//     S: Symbol,
//     H: BuildHasher,
// {
//     fn eq(&self, rhs: &Self) -> bool {
//         self.len() == rhs.len() && self.values == rhs.values
//     }
// }

impl Default for StringInterner<Sym, RandomState, 8> {
    #[inline]
    fn default() -> Self {
        StringInterner::<Sym, RandomState, 8>::new()
    }
}

// Should be manually cloned.
// See <https://github.com/Robbepop/string-interner/issues/9>.
// impl<S, H> Clone for StringInterner<S, H>
// where
//     S: Symbol,
//     H: Clone + BuildHasher,
// {
//     fn clone(&self) -> Self {
//         let values = self.values.clone();
//         let mut map = HashMap::with_capacity_and_hasher(values.len(), self.map.hasher().clone());
//         // Recreate `InternalStrRef` from the newly cloned `Box<str>`s.
//         // Use `extend()` to avoid `H: Default` trait bound required by `FromIterator for HashMap`.
//         map.extend(
//             values
//                 .iter()
//                 .enumerate()
//                 .map(|(i, s)| (InternalStrRef::from_str(s), S::from_usize(i))),
//         );
//         Self { values, map, hash_builder: self.hash_builder.clone() }
//     }
// }

// About `Send` and `Sync` impls for `StringInterner`
// --------------------------------------------------
//
// tl;dr: Automation of Send+Sync impl was prevented by `InternalStrRef`
// being an unsafe abstraction and thus prevented Send+Sync default derivation.
//
// These implementations are safe due to the following reasons:
//  - `InternalStrRef` cannot be used outside `StringInterner`.
//  - Strings stored in `StringInterner` are not mutable.
//  - Iterator invalidation while growing the underlying `Vec<Box<str>>` is prevented by
//    using an additional indirection to store strings.
unsafe impl<S, H, const N: usize> Send for StringInterner<S, H, N>
where
    S: Symbol + Send,
    H: BuildHasher,
{
}
unsafe impl<S, H, const N: usize> Sync for StringInterner<S, H, N>
where
    S: Symbol + Sync,
    H: BuildHasher,
{
}

impl<S, const N: usize> StringInterner<S, RandomState, N>
where
    S: Symbol,
{
    /// Creates a new empty `StringInterner`.
    #[inline]
    pub fn new() -> Self {
        StringInterner {
            map: array_init(|_| RwLock::new(HashMap::new())),
            values: array_init(|_| RwLock::new(Vec::new())),
            hash_builder: RandomState::new(),
        }
    }

    /// Creates a new `StringInterner` with the given initial capacity.
    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        StringInterner {
            map: array_init(|_| RwLock::new(HashMap::with_capacity(cap / N))),
            values: array_init(|_| RwLock::new(Vec::with_capacity(cap))),
            hash_builder: RandomState::new(),
        }
    }
}

impl<S, H, const N: usize> StringInterner<S, H, N>
where
    S: Symbol,
    H: BuildHasher + Clone,
{
    /// Index in arena from Sym
    pub fn sym_to_index(sym: S) -> usize {
        Symbol::to_usize(sym) & 0x00FFFFFF
    }

    /// Shard index from Sym
    pub fn sym_to_shard_index(sym: S) -> usize {
        Symbol::to_usize(sym) >> 24
    }

    /// Creates a new empty `StringInterner` with the given hasher.
    #[inline]
    pub fn with_hasher(hash_builder: H) -> Self {
        StringInterner {
            map: array_init(|_| RwLock::new(HashMap::with_hasher(hash_builder.clone()))),
            values: array_init(|_| RwLock::new(Vec::new())),
            hash_builder,
        }
    }

    /// Creates a new empty `StringInterner` with the given initial capacity and the given hasher.
    #[inline]
    pub fn with_capacity_and_hasher(cap: usize, hash_builder: H) -> Self {
        StringInterner {
            map: array_init(|_| RwLock::new(HashMap::with_capacity_and_hasher(cap / N, hash_builder.clone()))),
            values: array_init(|_| RwLock::new(Vec::with_capacity(cap))),
            hash_builder,
        }
    }

    fn hash(&self, s: &str) -> u64 {
        let mut hasher = self.hash_builder.build_hasher();
        hasher.write(s.as_bytes());
        hasher.finish()
    }

    fn shard_index(hash: u64) -> u64 {
        ((hash as usize) % N) as u64
    }

    /// Interns the given value.
    ///
    /// Returns a symbol to access it within this interner.
    ///
    /// This either copies the contents of the string (e.g. for str)
    /// or moves them into this interner (e.g. for String).
    #[inline]
    pub fn get_or_intern<T>(&self, val: T) -> S
    where
        T: Into<String> + AsRef<str>,
    {
        let shard_index = Self::shard_index(self.hash(val.as_ref()));
        let map = self.map[shard_index as usize].upgradable_read();
        match map.get(&val.as_ref().into()) {
            Some(e) => *e,
            None => self.intern_into_shard(shard_index, map, val),
        }
    }

    fn intern_into_shard<T>(&self, shard_index: u64, map_shard: RwLockUpgradableReadGuard<HashMap<InternalStrRef, S, H>>, new_val: T) -> S where T: Into<String> + AsRef<str> {
        let new_boxed_val: Arc<str> = new_val.into().into();
        let new_ref: InternalStrRef = new_boxed_val.as_ref().into();
        let new_id = {
            let mut values = self.values[shard_index as usize].write();
            let new_id: S = Self::make_symbol(shard_index, values.len());
            values.push(new_boxed_val);
            new_id
        };
        RwLockUpgradableReadGuard::upgrade(map_shard).insert(new_ref, new_id);
        new_id
    }

    /// Creates a new symbol for the current state of the interner.
    fn make_symbol(shard_index: u64, element_index: usize) -> S {
        S::from_usize(((shard_index as usize) << 24) | element_index)
    }

    /// Returns the string slice associated with the given symbol if available,
    /// otherwise returns `None`.
    #[inline]
    pub fn resolve(&self, symbol: S) -> Option<Arc<str>> {
        self.values[Self::sym_to_shard_index(symbol)].read().get(Self::sym_to_index(symbol)).cloned()
    }

    /// Returns the string associated with the given symbol.
    ///
    /// # Note
    ///
    /// This does not check whether the given symbol has an associated string
    /// for the given string interner instance.
    ///
    /// # Safety
    ///
    /// This will result in undefined behaviour if the given symbol
    /// had no associated string for this interner instance.
    #[inline]
    pub unsafe fn resolve_unchecked(&self, symbol: S) -> Arc<str> {
        self.values[Self::sym_to_shard_index(symbol)].read().get_unchecked(Self::sym_to_index(symbol)).clone()
    }

    /// Returns the symbol associated with the given string for this interner
    /// if existent, otherwise returns `None`.
    #[inline]
    pub fn get<T>(&self, val: T) -> Option<S>
    where
        T: AsRef<str>,
    {
        let shard_index = Self::shard_index(self.hash(val.as_ref()));
        self.map[shard_index as usize].read().get(&val.as_ref().into()).cloned()
    }

    /// Returns the number of uniquely interned strings within this interner.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.iter().map(RwLock::read).collect::<Vec<_>>().into_iter().rfold(0, |a, b| a+b.len())
    }

    /// Returns true if the string interner holds no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.iter().map(RwLock::read).collect::<Vec<_>>().into_iter().all(|v| v.is_empty())
    }

    /// Shrinks the capacity of the interner as much as possible.
    pub fn shrink_to_fit(&mut self) {
        self.map.iter().for_each(|m| m.write().shrink_to_fit());
        self.values.iter().for_each(|v| v.write().shrink_to_fit());
    }
}

// impl<T, S> FromIterator<T> for StringInterner<S>
// where
//     S: Symbol,
//     T: Into<String> + AsRef<str>,
// {
//     fn from_iter<I>(iter: I) -> Self
//     where
//         I: IntoIterator<Item = T>,
//     {
//         let iter = iter.into_iter();
//         let mut interner = StringInterner::with_capacity(iter.size_hint().0);
//         interner.extend(iter);
//         interner
//     }
// }

// impl<T, S> std::iter::Extend<T> for StringInterner<S>
// where
//     S: Symbol,
//     T: Into<String> + AsRef<str>,
// {
//     fn extend<I>(&mut self, iter: I)
//     where
//         I: IntoIterator<Item = T>,
//     {
//         for s in iter {
//             self.get_or_intern(s);
//         }
//     }
// }

///// Iterator over the pairs of associated symbols and interned strings for a `StringInterner`.
// pub struct Iter<'a, S> {
//     iter: iter::Enumerate<slice::Iter<'a, Arc<str>>>,
//     mark: marker::PhantomData<S>,
// }
//
// impl<'a, S> Iter<'a, S>
// where
//     S: Symbol + 'a,
// {
//     /// Creates a new iterator for the given StringIterator over pairs of
//     /// symbols and their associated interned string.
//     #[inline]
//     fn new<H>(interner: &'a StringInterner<S, H>) -> Self
//     where
//         H: BuildHasher,
//     {
//         Iter {
//             iter: interner.values.iter().enumerate(),
//             mark: marker::PhantomData,
//         }
//     }
// }
//
// impl<'a, S> Iterator for Iter<'a, S>
// where
//     S: Symbol + 'a,
// {
//     type Item = (S, &'a Arc<str>);
//
//     #[inline]
//     fn next(&mut self) -> Option<Self::Item> {
//         self.iter
//             .next()
//             .map(|(num, boxed_str)| (S::from_usize(num), boxed_str))
//     }
//
//     #[inline]
//     fn size_hint(&self) -> (usize, Option<usize>) {
//         self.iter.size_hint()
//     }
// }

/////Iterator over the interned strings of a `StringInterner`.
// pub struct Values<'a, S>
// where
//     S: Symbol + 'a,
// {
//     iter: slice::Iter<'a, Arc<str>>,
//     mark: marker::PhantomData<S>,
// }
//
// impl<'a, S> Values<'a, S>
// where
//     S: Symbol + 'a,
// {
//     /// Creates a new iterator for the given StringIterator over its interned strings.
//     #[inline]
//     fn new<H>(interner: &'a StringInterner<S, H>) -> Self
//     where
//         H: BuildHasher,
//     {
//         Values {
//             iter: interner.values.iter(),
//             mark: marker::PhantomData,
//         }
//     }
// }
//
// impl<'a, S> Iterator for Values<'a, S>
// where
//     S: Symbol + 'a,
// {
//     type Item = &'a Arc<str>;
//
//     #[inline]
//     fn next(&mut self) -> Option<Self::Item> {
//         self.iter.next()
//     }
//
//     #[inline]
//     fn size_hint(&self) -> (usize, Option<usize>) {
//         self.iter.size_hint()
//     }
// }
//
// impl<S, H> iter::IntoIterator for StringInterner<S, H>
// where
//     S: Symbol,
//     H: BuildHasher,
// {
//     type Item = (S, Arc<str>);
//     type IntoIter = IntoIter<S>;
//
//     fn into_iter(self) -> Self::IntoIter {
//         IntoIter {
//             iter: self.values.into_iter().enumerate(),
//             mark: marker::PhantomData,
//         }
//     }
// }
//
// /// Iterator over the pairs of associated symbol and strings.
// ///
// /// Consumes the `StringInterner` upon usage.
// pub struct IntoIter<S>
// where
//     S: Symbol,
// {
//     iter: iter::Enumerate<vec::IntoIter<Arc<str>>>,
//     mark: marker::PhantomData<S>,
// }
//
// impl<S> Iterator for IntoIter<S>
// where
//     S: Symbol,
// {
//     type Item = (S, Arc<str>);
//
//     fn next(&mut self) -> Option<Self::Item> {
//         self.iter
//             .next()
//             .map(|(num, boxed_str)| (S::from_usize(num), boxed_str))
//     }
//
//     #[inline]
//     fn size_hint(&self) -> (usize, Option<usize>) {
//         self.iter.size_hint()
//     }
// }
