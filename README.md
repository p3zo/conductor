# Conductor

Control music like a conductor.

## Usage

Run the application locally with `yarn && yarn start`.

Push changes to the `main` branch, then run `deploy.sh` to build the application on the `pages` branch where gh-pages
will serve it from.

## Future work

- Connect this app to a music generation model to enable a deeper level of control. For example, gestures could be
  mapped not just to model parameters like style and dynamics. Large and bouncy movements could tell the orchestra to
  play in a march style; small and lively movements could signal pizzicato.

- Train a model to replicate movement data captured in this application. An AI conductor trained in such a way could
  conduct a human orchestral ensemble to create a unique AI-led machine-human collaborative performance.

- Use gestures from drummers or dancers, rather than conductors.
