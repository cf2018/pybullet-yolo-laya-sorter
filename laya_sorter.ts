import { createInterface } from "node:readline";
import { Laya, type ChoiceQuestion } from "@receptron/laya";

type CupColor = "red" | "green" | "blue";
type BoxId = `${CupColor}_box`;

interface DecisionRequest {
  kind: "destination";
  cupColor: CupColor;
  hsvHue: number;
  colorScores: Record<CupColor, number>;
  visionConfidence: number;
}

interface VerificationRequest {
  kind: "verification";
  placements: Array<{
    cupColor: CupColor;
    boxId: string;
    placed: boolean;
  }>;
}

type WorkerRequest = DecisionRequest | VerificationRequest;

const expectedBox: Record<CupColor, BoxId> = {
  red: "red_box",
  green: "green_box",
  blue: "blue_box",
};

const destinationQuestion = {
  type: "choice",
  instructions:
    "Choose the dedicated box matching the measured cup color. Never send a cup to a differently colored box.",
  criteria: {
    red_box: "The red box, only for a red cup.",
    green_box: "The green box, only for a green cup.",
    blue_box: "The blue box, only for a blue cup.",
  },
} satisfies ChoiceQuestion;

const verificationQuestion = {
  type: "choice",
  instructions:
    "Based on the verification report, choose complete only when every cup is confirmed inside its matching-color box. Otherwise choose retry.",
  criteria: {
    complete: "All three cups are confirmed placed in their matching-color boxes.",
    retry: "At least one cup is not confirmed in its matching-color box; repeat the failed placement.",
  },
} satisfies ChoiceQuestion;

function isCupColor(value: unknown): value is CupColor {
  return value === "red" || value === "green" || value === "blue";
}

function isBoxId(value: string): value is BoxId {
  return value === "red_box" || value === "green_box" || value === "blue_box";
}

async function main(): Promise<void> {
  const laya = await Laya.load({
    sessionOptions: { intraOpNumThreads: 2 },
  });
  process.stdout.write(`${JSON.stringify({ ready: true })}\n`);

  const input = createInterface({ input: process.stdin, crlfDelay: Infinity });
  for await (const line of input) {
    try {
      const request = JSON.parse(line) as WorkerRequest;
      if (request.kind === "verification") {
        const physicallyComplete =
          request.placements.length === 3 &&
          request.placements.every(
            (placement) =>
              isCupColor(placement.cupColor) &&
              isBoxId(placement.boxId) &&
              placement.boxId === expectedBox[placement.cupColor] &&
              placement.placed,
          );
        const result = await laya.systemOne(
          {
            result: physicallyComplete
              ? "SUCCESS: all three cups are physically inside the matching red, green, and blue boxes."
              : "FAILURE: at least one cup is not confirmed inside its matching-color box. Retry the failed placement.",
            placements: request.placements.map((placement) => ({
              cup_color: placement.cupColor,
              box_id: placement.boxId,
              placed: placement.placed,
            })),
          },
          { completion: verificationQuestion },
        );
        const answer = result.answers.completion;
        const probability = answer.probabilities[answer.choice] ?? 0;
        process.stdout.write(
          `${JSON.stringify({
            kind: "verification",
            choice: answer.choice,
            probability,
            probabilities: answer.probabilities,
            confidence: answer.confidence,
            physicallyComplete,
          })}\n`,
        );
        continue;
      }
      if (request.kind !== "destination") {
        throw new Error("Unknown Laya request kind");
      }
      if (
        !isCupColor(request.cupColor) ||
        !Number.isFinite(request.hsvHue) ||
        !Number.isFinite(request.visionConfidence)
      ) {
        throw new Error("Invalid cup color decision request");
      }

      const result = await laya.systemOne(
        {
          cup: { color: request.cupColor },
          task: "Match cup color to bin color.",
        },
        { destination: destinationQuestion },
      );
      const answer = result.answers.destination;
      const destination = answer.choice;
      const probability = answer.probabilities[destination] ?? 0;
      const safeToSort =
        isBoxId(destination) &&
        destination === expectedBox[request.cupColor] &&
        probability >= 0.6;

      process.stdout.write(
        `${JSON.stringify({
          kind: "destination",
          destination,
          probability,
          confidence: answer.confidence,
          safeToSort,
        })}\n`,
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      process.stdout.write(`${JSON.stringify({ error: message })}\n`);
    }
  }

  await laya.close();
}

main().catch((error: unknown) => {
  const message = error instanceof Error ? error.stack ?? error.message : String(error);
  process.stderr.write(`${message}\n`);
  process.exitCode = 1;
});