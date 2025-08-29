# Risikovurdering

Denne risikovurderingen beskriver sentrale sikkerhetsaspekter og tiltak for Transcriber‑løsningen (FastAPI + ElevenLabs STT + Azure OpenAI + Outlook + valgfri Notion).

## 1. Nøkkel- og hemmelighetshåndtering
- Vektor: Uautorisert tilgang til API‑nøkler (ElevenLabs, Azure OpenAI, Microsoft, Notion).
- Konsekvens: Misbruk av tredjeparts‑API, økonomisk tap, datalekkasjer.
- Tiltak:
  - Oppbevar nøkler i `.env` som ikke sjekkes inn i git.
  - Rotér nøkler jevnlig, og ved mistanke om lekkasje.
  - Begrens rettigheter (minste privilegium) og aktiver brannmur/IP‑begrensning der mulig.
  - I prod: bruk hemmelighetshvelv (Azure Key Vault / AWS Secrets Manager) fremfor `.env`.

## 2. Datahåndtering og personvern
- Vektor: Lyd-/videofiler og transkripsjoner kan inneholde personopplysninger (GDPR).
- Konsekvens: Brudd på personvern, regulatoriske sanksjoner.
- Tiltak:
  - Minimer datalagring; slett midlertidige filer og mellomresultater.
  - Krypter data “in transit” (HTTPS). For prod: vurder “at rest” kryptering.
  - Informer brukere om formål og lagringstid; innhent samtykke der det er påkrevd.
  - Maskér/saniter sensitiv info før deling/lagring når mulig.

## 3. Tredjeparts‑overføringer (ElevenLabs/Azure/Notion/Microsoft)
- Vektor: Overføring av innhold til eksterne APIer.
- Konsekvens: Data på avveie hos tredjepart, avhengighet til eksterne SLA.
- Tiltak:
  - Bruk regionsinnstillinger som er i tråd med krav (EU/EØS hvis nødvendig).
  - Avklar databehandler‑avtaler (DPA) og les leverandørenes sikkerhetsdokumentasjon.
  - Loggfør kun aggregert/ufarlig meta‑info, ikke råinnhold.

## 4. Autentisering og autorisasjon (Microsoft Graph)
- Vektor: Feilkonfigurerte tillatelser (for vide), token‑tyveri.
- Konsekvens: Uautorisert lesing/skriving i kalenderdata.
- Tiltak:
  - Bruk kun nødvendige Graph‑scopes (User.Read, Calendars.ReadWrite).
  - Kortlevde tokens, og preferer “silent” flow med strenge opprinnelsesdomener.
  - Ikke logg access tokens; oppbevar kun i minne på klientsiden.

## 5. Rate limiting og DoS
- Vektor: Overdreven trafikk mot API eller tredjepartstjenester.
- Konsekvens: Tjenesten utilgjengelig; kostnadseksplosjon.
- Tiltak:
  - Innfør rate limiting (per IP/bruker) og backoff‑strategier.
  - Overvåk kvoter/kostnader hos tredjepart; varsling ved avvik.

## 6. Inputvalidering og robusthet
- Vektor: Ugyldige eller ondsinnede filer/forespørsler.
- Konsekvens: Feil, ressursforbruk, mulig sårbarhet.
- Tiltak:
  - Valider filtyper/størrelse; avvis tomme filer.
  - Tidsavbrudd på nettverkskall; håndter 4xx/5xx og gi trygge feilmeldinger.

## 7. Loggføring og innsyn
- Vektor: Logger kan inneholde sensitive data.
- Konsekvens: Utilsiktet eksponering.
- Tiltak:
  - Logg kun nødvendige meta‑data (status, lengder, suffikser), ikke innhold/hemmeligheter.
  - Slå av detaljert logging i prod, eller filtrer/saner før lagring.

## 8. Nettlesersikkerhet (frontend)
- Vektor: XSS/CSRF via brukerinput eller rendring av HTML.
- Konsekvens: Session/Token‑tyveri, manipulering av UI.
- Tiltak:
  - Escape all tekst når vi rendrer (som allerede gjort i UI).
  - Ikke lagre tokens i usikre lagre; MSAL håndterer session i minne.
  - Bruk `Content-Security-Policy` og `SameSite`‑cookies i prod.

## 9. Distribusjon og drift
- Vektor: Feil i miljøer (dev/test/prod), svak TLS, manglende oppdateringer.
- Konsekvens: Uautorisert tilgang, nedetid.
- Tiltak:
  - Skille miljøer med egne nøkler og konfig.
  - Prod: kjør bak reverse proxy med TLS (f.eks. Nginx/CloudFront) og health checks.
  - Patch avhengigheter jevnlig; `pip audit`/SCA‑verktøy i pipeline.

## 10. Endringshåndtering og versjonering
- Vektor: Uheldige endringer som påvirker sikkerhet.
- Konsekvens: Regressjon eller svekket sikkerhet.
- Tiltak:
  - Code review, CI‑tester og staging før prod.
  - Tagging/releases for sporbarhet; rollback‑plan.

## 11. Brukerfeil og UX‑risiko
- Vektor: Feil møtevalg, feil database, gamle miljøvariabler.
- Konsekvens: Data havner feil sted eller oppleves “borte”.
- Tiltak:
  - Tydelige feilmeldinger og “/debug”‑endepunkter (maskerte verdier).
  - UI som viser valgt møte/DB og oppdaterer status automatisk.

## Oppsummering av anbefalinger
- Hold hemmeligheter sikre og roter dem jevnlig.
- Minimer datalagring, bruk kryptering, avklar DPA og regionkrav.
- Stram inn Graph‑scopes og håndter tokens riktig.
- Innfør rate limiting og overvåk kostnader.
- Valider input, logg trygt og beskytt frontend‑rendring.
- Prod: kjør bak reverse proxy med TLS, oppdater avhengigheter, og bruk CI/CD med kontroller.
