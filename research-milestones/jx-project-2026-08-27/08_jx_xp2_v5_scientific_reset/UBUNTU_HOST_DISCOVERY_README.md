# JX-XP2 Ubuntu Host Discovery Review V1

This separately versioned package contains a private, read-only host-discovery collector and validator for a user-operated Ubuntu PC. It is not part of the operational ceremony DAG and grants no authority.

The collector requires no sudo, performs no installation, compilation, network access, GPU access, dynamics, simulation, registration, activation, publication, or artifact-content inspection. Its only explicit write is one caller-selected, absent report file created with no-follow exclusive creation at mode 0600. It does not overwrite, append, rename, unlink, or create temporary files.

Starting Python and reading its source can update access-time metadata on a relatime or strict-atime execution filesystem. Discovery V1 therefore makes the narrower and truthful claim `NO_EXPLICIT_PAYLOAD_WRITE_EXCEPT_EXCLUSIVE_PRIVATE_REPORT`; zero host metadata writes remain UNKNOWN unless noatime execution media or a pre-running trusted collector is established externally.

Run only after independently authenticating this self-free package inventory. Use an absolute, reviewed Python executable with isolated/no-site/no-bytecode flags:

`<ABSOLUTE_PYTHON3> -I -S -B <ABSOLUTE_PACKAGE_ROOT>/collect_host_discovery_v1.py --consent I_CONSENT_TO_PRIVATE_READ_ONLY_HOST_DISCOVERY_V1 --report <ABSOLUTE_ABSENT_REPORT_PATH> --persistent-ceremony-root <PATH> --ledger-root <PATH> --return-bundle-root <PATH> --staging-root <PATH> --final-content-root <PATH> --primary-runtime-c-root <PATH> --independent-runtime-d-root <PATH> --checkpoint-root <PATH> --output-root <PATH>`

Candidate paths may be absent. Absence or ambiguity yields UNKNOWN, never invented success. Checks that require mutation, privilege, isolation testing, toolchain execution, or ceremony evidence are forced to UNKNOWN. The overall report can be only FAIL or UNKNOWN; PASS is invalid in V1.

The report remains local/private. It cannot populate any external capability, policy, root-evidence, approval, ledger, registration, GO, activation, or simulation record. Transferring bytes to a different Ubuntu host cannot recreate the original nine retained `/tmp` root identities or listings.
