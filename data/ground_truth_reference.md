# Ground-truth reference for qa_gold.json

Use this to look up the correct answer for each item in `data/qa_gold.json`. All numbers are derived from `data/vendor_store.json` (source of truth) and `data/ground_truth.json` (anomaly labels).

## Vendor totals

| Vendor | # inv | Total spend | Mean | First | Last |
| --- | ---: | ---: | ---: | --- | --- |
| Acme Cloud Services | 5 | $4,131.27 | $826.25 | 2025-01-01 | 2025-05-04 |
| Blue Ridge Logistics | 5 | $7,227.96 | $1,445.59 | 2025-01-07 | 2025-02-16 |
| Corner Coffee Co. | 5 | $969.08 | $193.82 | 2025-01-01 | 2025-01-29 |
| Delta Print Shop | 5 | $2,081.13 | $416.23 | 2025-01-09 | 2025-03-19 |
| Eastwind Legal LLP | 5 | $26,955.27 | $5,391.05 | 2025-01-19 | 2025-05-23 |
| Foxtrot Facilities | 5 | $4,963.87 | $992.77 | 2025-01-05 | 2025-03-04 |
| Grove Web Studio | 5 | $11,868.95 | $2,373.79 | 2025-01-28 | 2025-06-03 |
| Harbor Insurance Brokers | 5 | $9,779.83 | $1,955.97 | 2025-01-03 | 2025-05-03 |
| Ivy Analytics | 5 | $19,007.39 | $3,801.48 | 2025-01-17 | 2025-05-10 |
| Juniper Office Supplies | 5 | $1,445.98 | $289.20 | 2025-01-07 | 2025-03-07 |
| Kestrel HR Partners | 5 | $13,727.23 | $2,745.45 | 2025-01-28 | 2025-06-02 |
| Larkspur Design | 5 | $9,314.50 | $1,862.90 | 2025-01-18 | 2025-03-30 |

## Vendors ranked by total spend

1. **Eastwind Legal LLP** — $26,955.27
2. **Ivy Analytics** — $19,007.39
3. **Kestrel HR Partners** — $13,727.23
4. **Grove Web Studio** — $11,868.95
5. **Harbor Insurance Brokers** — $9,779.83
6. **Larkspur Design** — $9,314.50
7. **Blue Ridge Logistics** — $7,227.96
8. **Foxtrot Facilities** — $4,963.87
9. **Acme Cloud Services** — $4,131.27
10. **Delta Print Shop** — $2,081.13
11. **Juniper Office Supplies** — $1,445.98
12. **Corner Coffee Co.** — $969.08

## Injected anomalies (ground truth)

These invoices were tagged as anomalous in the synthetic data. Any invoice NOT listed here is a normal invoice — the correct answer to "was it flagged?" is **no**.

| Invoice | Vendor | Injected category |
| --- | --- | --- |
| INV-00006 | Blue Ridge Logistics | amount_outlier |
| INV-00011 | Corner Coffee Co. | missing_po |
| INV-00014 | Corner Coffee Co. | off_cadence |
| INV-00031 | Grove Web Studio | amount_outlier |
| INV-00033 | Grove Web Studio | off_cadence |
| INV-00043 | Ivy Analytics | duplicate |
| INV-00048 | Juniper Office Supplies | duplicate |
| INV-00051 | Kestrel HR Partners | off_cadence |
| INV-00057 | Larkspur Design | amount_outlier |

## All invoices (per vendor)

Compact per-vendor list. `!` marks an injected anomaly.

### Acme Cloud Services

| Invoice | Date | Total | PO | Anomaly |
| --- | --- | ---: | --- | --- |
| INV-00001 | 2025-01-01 | $857.56 | - |  |
| INV-00002 | 2025-01-26 | $829.31 | - |  |
| INV-00003 | 2025-02-24 | $887.91 | - |  |
| INV-00004 | 2025-04-03 | $739.80 | - |  |
| INV-00005 | 2025-05-04 | $816.69 | - |  |

### Blue Ridge Logistics

| Invoice | Date | Total | PO | Anomaly |
| --- | --- | ---: | --- | --- |
| INV-00006 | 2025-01-07 | $1,633.95 | - | ! amount_outlier |
| INV-00007 | 2025-01-17 | $1,711.38 | PO-16006 |  |
| INV-00008 | 2025-01-30 | $1,477.17 | PO-46434 |  |
| INV-00009 | 2025-02-04 | $1,425.81 | PO-89840 |  |
| INV-00010 | 2025-02-16 | $979.65 | PO-17331 |  |

### Corner Coffee Co.

| Invoice | Date | Total | PO | Anomaly |
| --- | --- | ---: | --- | --- |
| INV-00011 | 2025-01-01 | $184.68 | PO-94259 | ! missing_po |
| INV-00012 | 2025-01-08 | $192.10 | PO-38746 |  |
| INV-00013 | 2025-01-16 | $218.80 | PO-30033 |  |
| INV-00014 | 2025-01-24 | $203.26 | PO-60019 | ! off_cadence |
| INV-00015 | 2025-01-29 | $170.24 | PO-99353 |  |

### Delta Print Shop

| Invoice | Date | Total | PO | Anomaly |
| --- | --- | ---: | --- | --- |
| INV-00016 | 2025-01-09 | $388.30 | - |  |
| INV-00017 | 2025-01-26 | $533.52 | - |  |
| INV-00018 | 2025-02-16 | $441.03 | - |  |
| INV-00019 | 2025-03-04 | $387.50 | - |  |
| INV-00020 | 2025-03-19 | $330.78 | - |  |

### Eastwind Legal LLP

| Invoice | Date | Total | PO | Anomaly |
| --- | --- | ---: | --- | --- |
| INV-00021 | 2025-01-19 | $6,143.84 | PO-26828 |  |
| INV-00022 | 2025-02-21 | $5,595.61 | PO-98039 |  |
| INV-00023 | 2025-03-21 | $3,412.04 | PO-39451 |  |
| INV-00024 | 2025-04-22 | $6,492.38 | - |  |
| INV-00025 | 2025-05-23 | $5,311.40 | PO-19287 |  |

### Foxtrot Facilities

| Invoice | Date | Total | PO | Anomaly |
| --- | --- | ---: | --- | --- |
| INV-00026 | 2025-01-05 | $1,038.83 | PO-22363 |  |
| INV-00027 | 2025-01-20 | $923.10 | - |  |
| INV-00028 | 2025-02-04 | $1,042.24 | PO-28373 |  |
| INV-00029 | 2025-02-16 | $1,011.40 | PO-82132 |  |
| INV-00030 | 2025-03-04 | $948.30 | - |  |

### Grove Web Studio

| Invoice | Date | Total | PO | Anomaly |
| --- | --- | ---: | --- | --- |
| INV-00031 | 2025-01-28 | $2,423.16 | - | ! amount_outlier |
| INV-00032 | 2025-03-05 | $1,239.26 | - |  |
| INV-00033 | 2025-04-16 | $2,088.02 | - | ! off_cadence |
| INV-00034 | 2025-05-06 | $3,072.78 | - |  |
| INV-00035 | 2025-06-03 | $3,045.73 | - |  |

### Harbor Insurance Brokers

| Invoice | Date | Total | PO | Anomaly |
| --- | --- | ---: | --- | --- |
| INV-00036 | 2025-01-03 | $1,676.10 | - |  |
| INV-00037 | 2025-01-27 | $2,205.82 | - |  |
| INV-00038 | 2025-02-23 | $1,694.09 | - |  |
| INV-00039 | 2025-03-28 | $1,959.39 | - |  |
| INV-00040 | 2025-05-03 | $2,244.43 | - |  |

### Ivy Analytics

| Invoice | Date | Total | PO | Anomaly |
| --- | --- | ---: | --- | --- |
| INV-00041 | 2025-01-17 | $5,083.82 | PO-30676 |  |
| INV-00042 | 2025-02-10 | $2,749.46 | PO-96951 |  |
| INV-00043 | 2025-03-10 | $3,944.16 | PO-45697 | ! duplicate |
| INV-00044 | 2025-04-05 | $3,649.88 | PO-22097 |  |
| INV-00045 | 2025-05-10 | $3,580.07 | PO-67912 |  |

### Juniper Office Supplies

| Invoice | Date | Total | PO | Anomaly |
| --- | --- | ---: | --- | --- |
| INV-00046 | 2025-01-07 | $356.71 | PO-86350 |  |
| INV-00047 | 2025-01-25 | $220.46 | PO-99399 |  |
| INV-00048 | 2025-02-08 | $327.63 | PO-64040 | ! duplicate |
| INV-00049 | 2025-02-24 | $232.66 | - |  |
| INV-00050 | 2025-03-07 | $308.52 | PO-50001 |  |

### Kestrel HR Partners

| Invoice | Date | Total | PO | Anomaly |
| --- | --- | ---: | --- | --- |
| INV-00051 | 2025-01-28 | $2,605.29 | - | ! off_cadence |
| INV-00052 | 2025-02-28 | $2,562.69 | - |  |
| INV-00053 | 2025-04-01 | $2,790.49 | - |  |
| INV-00054 | 2025-04-29 | $2,950.35 | - |  |
| INV-00055 | 2025-06-02 | $2,818.41 | - |  |

### Larkspur Design

| Invoice | Date | Total | PO | Anomaly |
| --- | --- | ---: | --- | --- |
| INV-00056 | 2025-01-18 | $2,549.16 | PO-19171 |  |
| INV-00057 | 2025-02-05 | $2,390.86 | PO-76469 | ! amount_outlier |
| INV-00058 | 2025-02-23 | $1,268.35 | PO-97153 |  |
| INV-00059 | 2025-03-18 | $1,213.84 | PO-47606 |  |
| INV-00060 | 2025-03-30 | $1,892.29 | PO-98555 |  |

## Meta

- Total invoices in the vendor store: **60**
- Earliest invoice date: **2025-01-01**
- Latest invoice date: **2025-06-03**
- Injected anomalies: **9**
- Distinct vendors: **12**

**Anything referring to 2024 or earlier does not exist in this data.**
