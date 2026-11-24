# Hardware references for four-qubit readout

Both layouts give each data transmon its own dispersive readout resonator. The dedicated layout assigns a two-port feedline to each resonator. The shared layout places all four resonators on one two-port feedline and distinguishes their responses by frequency. Within this comparison, the two layouts have eight and two readout reference planes, respectively, in addition to four XY drives.

## IBM Quantum hardware

Córcoles et al. demonstrated independent readout of four fixed-frequency transmons on a planar square lattice. Each qubit had its own readout resonator, microwave line, and JPA followed by a 4 K HEMT. The same chip lines also carried qubit control tones. The measured frequencies form close pairs at 6.491/6.494 GHz and 6.693/6.695 GHz. That packing was feasible with dedicated paths. A four-tone shared line needs enough separation to resolve each resonator and its state shift under the chosen pulse bandwidth. [Córcoles et al., Methods and Figs. 6–7](https://www.nature.com/articles/ncomms7979)

| IBM transmon | Qubit frequency (GHz) | Readout frequency (GHz) | Linewidth $\kappa/2\pi$ (MHz) | State separation $\lvert 2\chi\rvert/2\pi$ (MHz) |
| --- | ---: | ---: | ---: | ---: |
| Q1 | 5.303 | 6.494 | 0.615 | 3.0 |
| Q2 | 5.101 | 6.695 | 0.440 | 2.0 |
| Q3 | 5.291 | 6.491 | 0.287 | 2.5 |
| Q4 | 5.415 | 6.693 | 1.210 | 2.8 |

IBM's Falcon publications show how multiplexing was used in deployed hardware. Jurcevic et al. publish six multiplexed readout groups, R1–R6, for the 27-qubit Montreal processor. Nation et al. report 3:1 to 5:1 readout output multiplexing for Kolkata, with typical within-output frequency spacing of 50–60 MHz, mean cavity linewidth of 5.6 MHz, mean dispersive shift of 1.6 MHz in their convention, and 330 ns readout pulses. These are apparatus-specific measurements. [Jurcevic et al., Fig. 1(b)](https://arxiv.org/html/2008.08571#S2) · [Nation et al., Appendix B](https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.2.040326)

IBM's public Heron description gives the qubit connectivity and signal delivery technology. `IBMBackend` describes `meas_map` for backends that expose multiplexed measurement groups, and calibration records can include readout error, conditional assignment errors, readout duration, and coherence. Use those fields to identify documented scheduling groups and result metrics when present. Physical feedline routes, filters, resonator linewidths, port reference planes, and scattering parameters for a Heron device would require additional device-specific data. [IBM processor types](https://quantum.cloud.ibm.com/docs/en/ibm-quantum-compute/processor-types#heron) · [IBMBackend `meas_map`](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/ibm-backend#meas_map) · [IBM backend details](https://quantum.cloud.ibm.com/docs/en/ibm-quantum-compute/qpu-information)

## AWS Center for Quantum Computing hardware

The AWS dual-rail erasure-qubit experiment used three transmons. Each had its own readout resonator attached to the same transmission line. Q1 and Q3 had additional Purcell filters. The 170 and 238 MHz spacings between the three measured resonances bracket the 200 MHz design spacing for EmulateQPU-4Q. The AWS circuit used two transmons as a logical dual rail and the third as a readout ancilla. Its absolute readout band, 7.341–7.749 GHz, sits above the local 6.45–7.05 GHz targets. [Levine et al., Appendix A and Table I](https://journals.aps.org/prx/pdf/10.1103/PhysRevX.14.011051)

| AWS transmon | Qubit idle frequency (GHz) | Readout frequency (GHz) | Linewidth $\kappa_{\rm RO}/2\pi$ (MHz) | Dispersive shift $\chi_{\rm RO}/2\pi$ (MHz, as reported) |
| --- | ---: | ---: | ---: | ---: |
| Q1 | 5.10 | 7.749 | 9.3 ± 0.4 | 3.73 ± 0.07 |
| Q2 | 5.10 | 7.511 | 0.87 ± 0.02 | 0.32 ± 0.01 |
| Q3 (ancilla) | 3.74 | 7.341 | 6.7 ± 0.2 | 2.53 ± 0.03 |

AWS's Ocelot prototype stores data in cat oscillators and uses transmon ancillas for syndrome measurements. Its paper documents ancilla groups of two or three per multiplexed line. Each ancilla's resonator is inductively coupled to its feedline through a single-pole quarter-wave Purcell filter, with reported resonator decay rates $\kappa_r/2\pi$ above 5 MHz. This is evidence for filtered multiplexed ancilla readout in a different kind of processor. [Amazon's Ocelot description](https://www.amazon.science/blog/amazon-announces-ocelot-quantum-chip) · [Putterman et al., supplementary sections B and F.3](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-025-08642-7/MediaObjects/41586_2025_8642_MOESM1_ESM.pdf)

Amazon Braket is a cloud service whose catalog includes third-party QPUs. `GetDevice` returns provider-specific capabilities, including gate connectivity, supported operations, and calibration fields where supplied. Pulse-enabled providers can also expose named ports and carrier frames. Treat these as controller interface metadata. An analog reconstruction needs the on-chip network and package, along with measured resonator and transmission parameters. [Braket device catalog](https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices.html) · [GetDevice API](https://docs.aws.amazon.com/braket/latest/APIReference/API_GetDevice.html) · [Braket Pulse](https://docs.aws.amazon.com/braket/latest/developerguide/braket-pulse-control.html)

## What to match in EmulateQPU-4Q

The nominal 6.45, 6.65, 6.85, and 7.05 GHz targets span 600 MHz. The IBM dedicated device anchors the low end of that band. An ETH Zürich shared-line experiment reported fitted resonator frequencies of 6.409–7.214 GHz and effective linewidths of 3.1–14.3 MHz across its five measured resonator–filter pairs. Those ranges support using the same four design targets as an initial condition for both layouts. The 200 MHz spacing is wider than Kolkata's reported 50–60 MHz. It trades a wider control and amplification band for more separation between neighboring tones. [Heinsoo et al., Fig. 2 and Table I](https://arxiv.org/pdf/1801.07904)

| Hold fixed in the first comparison | Extract or measure for each topology |
| --- | --- |
| Four data qubits, three couplers, and the same idle bias point | The dressed qubit frequencies and readout mode assignment under each qubit's ground and excited states |
| Four nominal resonator frequencies and identical nominal qubit pickup and CPW dimensions | The realized resonator frequencies, loaded linewidths, external quality factors, and state splitting $2\chi$ |
| Substrate, enclosure, chip-plane calibration convention, and a common reference impedance | Full complex scattering versus frequency, including both directions, both reflections, off-path transmission on dedicated lines, and the aggregate shared-line response |
| Measurement pulses and their chip-plane calibration convention | Single- and four-tone responses, spectator shifts and dephasing, resonator ringdown, and the response of the complete signal chain |

Use the ground- and excited-state frequency difference directly when comparing studies with different $\chi$ conventions. Route changes can alter the external linewidth, background phase, and cross-coupling even with identical nominal resonators. First characterize each physical layout as drawn. For a comparison at equal cavity loading and drive strength, match the extracted linewidth and calibrated intracavity population or specify their differences. Report the state-dependent output separation separately. A linear field solve can establish port response and supported modes. Assignment fidelity and qubit-state backaction require a state-dependent model and calibration or measurement data.
