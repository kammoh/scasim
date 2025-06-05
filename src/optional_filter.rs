pub trait FilterAdaptorWithCondition: Iterator {
    fn opt_filter<P>(
        self,
        predicate: P,
        condition: bool,
    ) -> OptionalFilter<
        std::iter::Chain<
            std::iter::Flatten<std::option::IntoIter<Self>>,
            std::iter::Flatten<std::option::IntoIter<std::iter::Filter<Self, P>>>,
        >,
    >
    where
        Self: Sized,
        P: FnMut(&Self::Item) -> bool,
    {
        let (unfiltered, filtered) = if condition {
            (None, Some(self.filter(predicate)))
        } else {
            (Some(self), None)
        };

        let new_iter = unfiltered
            .into_iter()
            .flatten()
            .chain(filtered.into_iter().flatten());

        OptionalFilter::new(new_iter)
    }
}

impl<T: Iterator> FilterAdaptorWithCondition for T {}

pub trait OptFilterAdaptor: Iterator {
    fn opt_filter_option<P>(
        self,
        predicate: Option<P>,
    ) -> OptionalFilter<
        std::iter::Chain<
            std::iter::Flatten<std::option::IntoIter<Self>>,
            std::iter::Flatten<std::option::IntoIter<std::iter::Filter<Self, P>>>,
        >,
    >
    where
        Self: Sized,
        P: FnMut(&Self::Item) -> bool,
    {
        let (unfiltered, filtered) = if let Some(predicate) = predicate {
            (None, Some(self.filter(predicate)))
        } else {
            (Some(self), None)
        };

        let new_iter = unfiltered
            .into_iter()
            .flatten()
            .chain(filtered.into_iter().flatten());

        OptionalFilter::new(new_iter)
    }
}

impl<T: Iterator> OptFilterAdaptor for T {}

// pub trait OptFilterMapAdaptor: Iterator {
//     fn opt_filter_map<P>(
//         self,
//         f: Option<P>,
//     ) -> OptionalFilter<
//         std::iter::Chain<
//             std::iter::Flatten<std::option::IntoIter<Self>>,
//             std::iter::Flatten<std::option::IntoIter<std::iter::FilterMap<Self, P>>>,
//         >,
//     >
//     where
//         Self: Sized,
//         P: FnMut(Self::Item) -> Option<Self::Item>,
//     {
//         let (unfiltered, filtered) = if let Some(predicate) = f {
//             (None, Some(self.filter_map(predicate)))
//         } else {
//             (Some(self), None)
//         };

//         let new_iter = unfiltered
//             .into_iter()
//             .flatten()
//             .chain(filtered.into_iter().flatten());

//         OptionalFilter::new(new_iter)
//     }
// }

// impl<T: Iterator> OptFilterMapAdaptor for T {}

pub struct OptionalFilter<I> {
    iter: I,
}

impl<I> OptionalFilter<I> {
    fn new(iter: I) -> OptionalFilter<I> {
        OptionalFilter { iter }
    }
}

impl<I: Iterator> Iterator for OptionalFilter<I> {
    type Item = I::Item;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}
