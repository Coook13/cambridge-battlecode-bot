# Cambridge Battlecode Bot

My bot iterations from Cambridge Battlecode 2026, a real-time-strategy programming
competition run on Cambridge's `cambc` game engine. Teams write a bot in Python that
controls units — miners, builders, raiders, turrets — under a shared economy and combat
model, then battle other teams' bots on a fixed map.

I played this as part of a small team. This repo holds the bot versions I personally
authored: each folder is one hypothesis about strategy — economy timing, unit
composition, aggression windows — tested against the field and against teammates' bots
in scrimmage. The naming (`v13`, `v54`, `v72`...) tracks the actual iteration order.

The competition has concluded.

## Structure

Each `bots/vN_<name>/` folder is a self-contained bot: entry point, unit controllers
(miner/builder/raider/turret), a shared local-map/pathfinding layer, and an economy
state tracker. Later versions build on earlier ones as the strategy sharpened round to
round — you can see the progression from early economy-only builds toward the
aggression and counter-play versions in the 70s–90s.

## Running a bot

Battlecode bots run inside Cambridge's `cambc` CLI, which is not part of this repo
(it's the competition's own engine, downloaded separately). Point `cambc` at any
`bots/vN_.../` folder as a player to run or replay a match.
