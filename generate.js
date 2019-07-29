const { Draco, Data, Result, Witness } = require('ndraco-core');
const data = require('./sunshine.json');
const fs = require('fs');

const dataObj = Data.fromArray(data);

const program = 'view(v1).\n' + dataObj.asp;

const output = [];

for (let i = 0; i < 50; i += 1) {
  const result = Draco.run(program, { optimize: false, models: 1, randomFreq: 1, randomSeed: i});
  const witnesses = Result.toWitnesses(result);
  for (const wit of witnesses) {
    const dict = Witness.toVegaLiteSpecDictionary(wit);
    output.push({
      draco: wit.facts,
      vl: dict['v1']
    });
  };
}

fs.writeFileSync('./examples.json', JSON.stringify(output, null, 2));