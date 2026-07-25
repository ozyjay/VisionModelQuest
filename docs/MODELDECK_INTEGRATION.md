# ModelDeck integration direction

No ModelDeck code or configuration is changed by this project.

After direct adapters are qualified, an optional provider should send the identical curated
workload only to a published ModelDeck gateway route. It can then compare direct runtime
latency, worker latency, gateway overhead, lifecycle recovery and scheduler behaviour.
The integration must:

- use the stable loopback gateway and public route contract;
- never call private worker ports;
- require an explicit published route and never create or rebind one;
- preserve fixture/question IDs and the normal report privacy boundary;
- capture the exact ModelDeck version, route contract and worker evidence;
- restore the initial worker state after comparison.

Any runtime-manifest or worker-adapter work discovered by that comparison must be proposed
for separate review in ModelDeck. VisionModelQuest must not modify it automatically.

