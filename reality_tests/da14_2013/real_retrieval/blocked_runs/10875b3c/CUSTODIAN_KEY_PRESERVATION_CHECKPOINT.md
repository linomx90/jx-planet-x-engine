# JX Reality Test 1 — Custodian-Key Preservation Checkpoint

**Date:** 2026-09-02 UTC  
**Status:** `ORIGINAL_CUSTODIAN_PRIVATE_KEY_NOT_PRESERVED`  
**Affected public-key fingerprint:** `8211e6a91fd5aca36fb349a4c41d9499ee2b6f90c86f084c19aa2f9f6cb0785f`

The original encrypted RSA-3072 private key and its password were generated in a temporary working environment for the first DA14 real-retrieval attempt. They were not uploaded to persistent custody before that temporary environment reset, and they are no longer recoverable from the active workspace.

No real DA14 source request occurred during the attempt. No training data, encrypted holdout, or encrypted quarantine was created under that key. Therefore, no observation data was lost or made permanently unreadable.

The committed public key must not be used for a later retrieval because its matching private key is unavailable. Before any prospectively authorized second attempt, a replacement keypair must be generated and its custody must be verified in this order:

1. create an encrypted private key and separate password;
2. verify decryption locally using a synthetic record;
3. archive the encrypted key and password in persistent custody before retrieval;
4. freeze the replacement public-key fingerprint and input hashes;
5. run the retrieval environment with the public key only.

This checkpoint does not authorize another retrieval attempt.
