use ayto_random::{Event, MatchingCeremony, Pair, TruthBooth};
use std::time::SystemTime;

fn main() {
    let females = vec![
        "Aline",     // 0
        "Ivana",     // 1
        "Katharina", // 2
        "Laura",     // 3
        "Luisa",     // 4
        "Madleine",  // 5
        "Melissa",   // 6
        "Michelle",  // 7
        "Nadine",    // 8
        "Sabrina",   // 9
    ];
    let males = vec![
        "Aleks",   // 0
        "Axel",    // 1
        "Dominic", // 2
        "Edin",    // 3
        "Elisha",  // 4
        "Ferhat",  // 5
        "Juliano", // 6
        "Kevin",   // 7
        "Laurin",  // 8
        "Mo",      // 9
        "René",    // 10
    ];
    let events = vec![
        Event::TruthBooth(TruthBooth::new(Pair::new(1, 9), false)),
        Event::TruthBooth(TruthBooth::new(Pair::new(6, 2), false)),
        Event::TruthBooth(TruthBooth::new(Pair::new(0, 9), true)),
        Event::TruthBooth(TruthBooth::new(Pair::new(6, 8), true)),
        Event::TruthBooth(TruthBooth::new(Pair::new(1, 4), false)),
        Event::TruthBooth(TruthBooth::new(Pair::new(7, 10), true)),
        // Matching Ceremony 1
        Event::MatchingCeremony(MatchingCeremony::new(
            vec![
                Pair::new(0, 7),
                Pair::new(1, 6),
                Pair::new(2, 10),
                Pair::new(3, 5),
                Pair::new(4, 1),
                Pair::new(5, 8),
                Pair::new(6, 2),
                Pair::new(7, 9),
                Pair::new(8, 4),
                Pair::new(9, 0),
            ],
            1,
        )),
        // Matching Ceremony 2
        Event::MatchingCeremony(MatchingCeremony::new(
            vec![
                Pair::new(0, 5),
                Pair::new(1, 4),
                Pair::new(2, 7),
                Pair::new(3, 6),
                Pair::new(4, 10),
                Pair::new(5, 1),
                Pair::new(6, 8),
                Pair::new(7, 9),
                Pair::new(8, 0),
                Pair::new(9, 2),
            ],
            2,
        )),
        // Matching Ceremony 3
        Event::MatchingCeremony(MatchingCeremony::new(
            vec![
                Pair::new(0, 9),
                Pair::new(1, 4),
                Pair::new(2, 7),
                Pair::new(3, 6),
                Pair::new(4, 10),
                Pair::new(5, 1),
                Pair::new(6, 8),
                Pair::new(7, 5),
                Pair::new(8, 0),
                Pair::new(9, 2),
            ],
            3,
        )),
        // Matching Ceremony 4
        Event::MatchingCeremony(MatchingCeremony::new(
            vec![
                Pair::new(0, 9),
                Pair::new(1, 1),
                Pair::new(2, 6),
                Pair::new(3, 7),
                Pair::new(4, 10),
                Pair::new(5, 5),
                Pair::new(6, 8),
                Pair::new(7, 4),
                Pair::new(8, 2),
                Pair::new(9, 0),
            ],
            2,
        )),
        // Matching Ceremony 5
        Event::MatchingCeremony(MatchingCeremony::new(
            vec![
                Pair::new(0, 9),
                Pair::new(1, 2),
                Pair::new(2, 7),
                Pair::new(3, 6),
                Pair::new(4, 5),
                Pair::new(5, 0),
                Pair::new(6, 8),
                Pair::new(7, 1),
                Pair::new(8, 10),
                Pair::new(9, 4),
            ],
            5,
        )),
        // Matching Ceremony 6
        Event::MatchingCeremony(MatchingCeremony::new(
            vec![
                Pair::new(0, 9),
                Pair::new(1, 6),
                Pair::new(2, 7),
                Pair::new(3, 3),
                Pair::new(4, 1),
                Pair::new(5, 0),
                Pair::new(6, 8),
                Pair::new(7, 10),
                Pair::new(8, 4),
                Pair::new(9, 2),
            ],
            5,
        )),
    ];
    let game = ayto_random::Game::new(females, males, events);

    let now = SystemTime::now();

    // Calculate and print the probabilities.
    let probability_map = game.calculate_chances(20_000_000);
    let mut probabilities: Vec<_> = probability_map.into_iter().collect();
    probabilities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (i, (pair, prob)) in probabilities.iter().enumerate() {
        println!("{}: {:?} - {:.4}", i, pair, prob);
    }

    let elapsed = now.elapsed().unwrap();
    println!("Elapsed time: {:?}", elapsed);
}
