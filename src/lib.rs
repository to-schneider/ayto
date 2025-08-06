//! # AYTO Solver Library
//!
//! This library provides the core logic for simulating and calculating probabilities for the
//! reality TV show "Are You The One?". It uses a Monte Carlo method to estimate the chances of each
//! potential couple being a "perfect match" based on known events like Truth Booths and Matching
//! Ceremonies.

use std::collections::HashMap;
use rand::Rng;
use rand::prelude::SliceRandom;
use rayon::prelude::*;

// --- Core Type Definitions ---

/// Represents a potential couple, defined by the numerical index of a female and a male contestant.
///
/// The struct derives several traits to enable comparisons and sorting (`Eq`, `Ord`, etc.). This is
/// crucial for efficiently checking for matches later in the simulation, as it allows for
/// algorithms like binary search and fast intersection counting on sorted lists.
#[derive(Eq, Ord, PartialEq, PartialOrd)]
pub struct Pair {
    /// The index of the female contestant in the original list.
    female_idx: usize,
    /// The index of the male contestant in the original list.
    male_idx: usize,
}

impl Pair {
    /// Constructs a new `Pair`.
    ///
    /// # Arguments
    ///
    /// * `female_idx` - The index of the female contestant.
    /// * `male_idx` - The index of the male contestant.
    ///
    /// # Returns
    ///
    /// A new `Pair` instance containing the provided indices.
    pub fn new(female_idx: usize, male_idx: usize) -> Self {
        Pair {
            female_idx,
            male_idx,
        }
    }
}

// --- Game Event Definitions ---

/// Represents the outcome of a single trip to the Truth Booth.
///
/// A Truth Booth definitively confirms or denies if a single `Pair` is a perfect match.
pub struct TruthBooth {
    /// The pair that went into the booth.
    pair: Pair,
    /// The result: `true` if it was a perfect match, `false` otherwise.
    is_match: bool,
}

impl TruthBooth {
    /// Constructs a new `TruthBooth` event.
    ///
    /// # Arguments
    ///
    /// * `pair` - The `Pair` that was sent to the Truth Booth.
    /// * `is_match` - A boolean indicating if the pair was confirmed as a perfect match.
    ///
    /// # Returns
    ///
    /// A new `TruthBooth` instance.
    pub fn new(pair: Pair, is_match: bool) -> Self {
        TruthBooth { pair, is_match }
    }
}

/// Represents the outcome of a Matching Ceremony.
///
/// At a Matching Ceremony, the contestants form a complete matching, and they are told how many of
/// their chosen pairs are correct, but not *which* ones.
pub struct MatchingCeremony {
    /// The specific matching that the contestants chose for the ceremony. This vector should be
    /// sorted before being stored to optimize later comparisons.
    matching: Vec<Pair>,
    /// The number of perfect matches in that matching.
    num_correct: usize,
}

impl MatchingCeremony {
    /// Constructs a new `MatchingCeremony` event.
    ///
    /// # Arguments
    ///
    /// * `matching` - A `Vec<Pair>` representing the couples chosen at the ceremony. The vector
    ///   will be sorted internally.
    /// * `num_correct` - The number of correct matches (beams of light) for the given matching.
    ///
    /// # Returns
    ///
    /// A new `MatchingCeremony` instance.
    pub fn new(mut matching: Vec<Pair>, num_correct: usize) -> Self {
        matching.sort_unstable();
        MatchingCeremony {
            matching,
            num_correct,
        }
    }
}

/// An enum representing a known event that provides a clue to the true solution.
///
/// This enum consolidates all types of game events (Truth Booths, Matching Ceremonies) into a
/// single type, making it easier to process them generically.
pub enum Event {
    /// A Truth Booth result.
    TruthBooth(TruthBooth),
    /// A Matching Ceremony result.
    MatchingCeremony(MatchingCeremony),
}

impl Event {
    /// Checks if a potential `candidate_matching` is consistent with this event's outcome.
    ///
    /// This is the core logic for filtering simulations. A candidate matching is only considered
    /// "possible" if it is consistent with *all* known events from the show.
    ///
    /// # Arguments
    ///
    /// * `candidate_matching` - A complete, sorted matching to check against the event.
    ///
    /// # Returns
    ///
    /// Returns `true` if the candidate matching could be the true solution given this event,
    /// otherwise returns `false`.
    fn is_consistent_with(&self, candidate_matching: &Vec<Pair>) -> bool {
        match self {
            Event::TruthBooth(tb) => {
                // For a Truth Booth, we check if the candidate matching contains the booth's pair.
                // A binary search is very efficient here because `candidate_matching` is sorted.
                // The result is consistent if our finding matches the truth booth's official
                // result.
                let pair_found = candidate_matching.binary_search(&tb.pair).is_ok();
                pair_found == tb.is_match
            }

            Event::MatchingCeremony(mc) => {
                // For a Matching Ceremony, we count how many pairs from the ceremony's matching
                // also appear in the candidate matching. This uses an efficient "two-pointer" or
                // "merge join" algorithm to find the intersection size of two sorted lists in a
                // single pass.
                let mut candidate_iter = candidate_matching.iter().peekable();
                let mut ceremony_iter = mc.matching.iter().peekable();
                let mut common_pairs = 0;

                // Loop as long as both lists have elements to compare.
                while let (Some(candidate_pair), Some(ceremony_pair)) =
                    (candidate_iter.peek(), ceremony_iter.peek())
                {
                    match candidate_pair.cmp(ceremony_pair) {
                        // The pair in the candidate list is "smaller" than the one in the ceremony
                        // list. It can't match the current ceremony pair or any later ones, so we
                        // move to the next candidate pair.
                        std::cmp::Ordering::Less => {
                            candidate_iter.next();
                        }

                        // The pair in the ceremony list is smaller. Advance the ceremony iterator.
                        std::cmp::Ordering::Greater => {
                            // The ceremony pair is smaller. Advance it.
                            ceremony_iter.next();
                        }

                        // The pairs are identical. We found a match.
                        std::cmp::Ordering::Equal => {
                            candidate_iter.next();
                            ceremony_iter.next();
                            common_pairs += 1;
                        }
                    }
                }

                // The candidate is consistent only if the number of common pairs found is exactly
                // the number of correct pairs indicated by the ceremony.
                common_pairs == mc.num_correct
            }
        }
    }
}

// --- Main Game Structure ---

/// Represents the entire state of the "Are You The One?" game.
///
/// It holds the list of contestants and all the known events that have occurred so far in the
/// season.
pub struct Game<'a> {
    /// A list of names for the female contestants.
    females: Vec<&'a str>,
    /// A list of names for the male contestants.
    males: Vec<&'a str>,
    /// A list of all `TruthBooth` and `MatchingCeremony` events that have occurred.
    events: Vec<Event>,
}

impl<'a> Game<'a> {
    /// Creates a new game instance.
    ///
    /// # Arguments
    ///
    /// * `females` - A vector of string slices representing the names of the female contestants.
    /// * `males` - A vector of string slices representing the names of the male contestants.
    /// * `events` - A vector of `Event` enums that have occurred in the game.
    ///
    /// # Returns
    ///
    /// A new `Game` instance initialized with the provided contestants and events.
    pub fn new(females: Vec<&'a str>, males: Vec<&'a str>, events: Vec<Event>) -> Self {
        Game {
            females,
            males,
            events,
        }
    }

    /// Runs a Monte Carlo simulation to estimate match probabilities.
    ///
    /// This method performs the main work. It generates a large number of random but valid
    /// matchings. Each random matching is tested against all known `events`. Those that are
    /// consistent with all events are considered "possible scenarios". The probability for any
    /// given pair is then estimated as the fraction between the count of each pair in all possible
    /// scenarios and the total number of possible scenarios found.
    ///
    /// This process is parallelized using Rayon for significant speed improvements.
    ///
    /// # Arguments
    ///
    /// * `num_simulations` - The number of random matchings to generate and test. A higher number
    ///   (e.g., > 1,000,000) yields more accurate probabilities but takes longer.
    ///
    /// # Returns
    ///
    /// A 2D vector (`Vec<Vec<f64>>`) where `probabilities[i][j]` is the estimated probability of
    /// `females[i]` and `males[j]` being a perfect match.
    pub fn calculate_chances(&self, num_simulations: usize) -> HashMap<(String, String), f64> {
        let n_females = self.females.len();
        let n_males = self.males.len();

        // The core of the simulation, running in parallel.
        let (occurrences, n_possible_matchings) = (0..num_simulations)
            .into_par_iter()
            // The `fold` operation gives each thread its own local accumulator. This avoids
            // expensive synchronization on every single simulation. Each thread gets a tuple: (a 2D
            // vector for counts, a counter for valid matchings).
            .fold(
                || (vec![vec![0; n_males]; n_females], 0),
                |mut thread_result, _| {
                    // Create a new random number generator for this thread.
                    let mut rng = rand::rng();

                    // Generate one random, valid matching.
                    let random_matching = generate_random_matching(n_females, n_males, &mut rng);

                    // Check if this random matching is consistent with all known events.
                    let is_possible = self
                        .events
                        .iter()
                        .all(|event| event.is_consistent_with(&random_matching));

                    // If it's a possible scenario, update this thread's local results.
                    if is_possible {
                        let (occurrences, n_possible_matchings) = &mut thread_result;
                        *n_possible_matchings += 1;
                        // Increment the count for each pair present in this valid matching.
                        for pair in random_matching {
                            occurrences[pair.female_idx][pair.male_idx] += 1;
                        }
                    }
                    thread_result
                },
            )
            // The `reduce` operation combines the results from all threads into a single final
            // result.
            .reduce(
                || (vec![vec![0; n_males]; n_females], 0),
                |mut total_result, thread_result| {
                    // Unpack the total and thread-local results.
                    let (total_occurrences, total_n_possible_matchings) = &mut total_result;
                    let (thread_occurrences, thread_n_possible_matchings) = thread_result;

                    // Add the counts from this thread to the grand total.
                    for i in 0..n_females {
                        for j in 0..n_males {
                            total_occurrences[i][j] += thread_occurrences[i][j];
                        }
                    }
                    *total_n_possible_matchings += thread_n_possible_matchings;
                    total_result
                },
            );

        // If no valid matchings were found (either due to low simulation count or contradictory
        // events), we return an empty map.
        if n_possible_matchings == 0 {
            return HashMap::new();
        }

        // Convert the raw occurrence counts into probabilities by dividing by the total number of
        // valid scenarios found.
        let probabilities = occurrences
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|count| count as f64 / n_possible_matchings as f64)
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>();
        
        // Create a map of (female, male) pairs to their corresponding probability.
        let mut probability_map = HashMap::new();
        for (i, row) in probabilities.iter().enumerate() {
            for (j, prob) in row.iter().enumerate() {
                let key = (self.females[i].into(), self.males[j].into());
                probability_map.insert(key, *prob);
            }       
        }

        probability_map
    }
}

/// Generates a single, random matching, handling unequal numbers of contestants.
///
/// This function creates a set of pairs for a simulation round. If the number of females and males
/// is different, it ensures everyone in the larger group gets a partner by randomly duplicating
/// contestants from the smaller group.
///
/// The logic extends the list of contestants from the smaller group by adding
/// `larger_group_size - smaller_group_size` randomly selected indices from that same smaller
/// group. The two lists are then paired to produce a final matching whose size is equal to that of
/// the larger original group.
///
/// # Arguments
///
/// * `n_females` - The total number of female contestants.
/// * `n_males` - The total number of male contestants.
/// * `rng` - A mutable reference to a random number generator.
///
/// # Returns
///
/// A `Vec<Pair>` representing one complete, random matching. The vector is sorted, and its length
/// will be equal to `max(n_females, n_males)`.
fn generate_random_matching(n_females: usize, n_males: usize, rng: &mut impl Rng) -> Vec<Pair> {
    // Create lists of indices for both groups.
    let mut females: Vec<usize> = (0..n_females).collect();
    let mut males: Vec<usize> = (0..n_males).collect();

    // Handle cases where the number of contestants is unequal. The goal is to pad the smaller list
    // by duplicating its members until its length matches the larger one.
    if n_females < n_males {
        // Calculate the difference in size.
        let diff = n_males - n_females;
        // Extend the list of females by adding random duplicates from the original set. This
        // ensures every male will have a female partner to be paired with.
        females.extend((0..diff).map(|_| rng.random_range(0..n_females)));
    }
    // Handle the case where there are more (or an equal number of) females than males.
    else {
        // Calculate the difference in size.
        let diff = n_females - n_males;
        // Extend the list of males by adding random duplicates from the original set.
        males.extend((0..diff).map(|_| rng.random_range(0..n_males)));
    }

    // Shuffle the male indices to create a random permutation for pairing.
    males.shuffle(rng);

    // Create pairs by zipping the (potentially extended) female and male lists. The `zip` operation
    // will stop when the shorter of the two lists is exhausted. Because we padded the originally
    // smaller list, the final number of pairs will equal the size of the originally larger group.
    let mut matching: Vec<Pair> = females
        .iter()
        .zip(males.iter())
        .map(|(&f, &m)| Pair {
            female_idx: f,
            male_idx: m,
        })
        .collect();

    // Sort the final list of pairs. This is critical for efficient processing (e.g., binary
    // searches, intersection checks) later in the simulation.
    matching.sort();
    matching
}
