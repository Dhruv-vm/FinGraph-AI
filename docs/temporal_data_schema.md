# FinGraph AI — Temporal Data Schema

## Design Principle

FinGraph AI represents financial information as time-dependent observations.

Every document, event, relationship, market observation, sentiment record, and prediction must have sufficient temporal metadata to determine whether information was available at a given prediction timestamp.

## Core Entities

### Companies

```text
company_id
ticker
name
sector
cik
created_at
