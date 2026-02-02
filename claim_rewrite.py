import re

class ClaimRewriter:

    def rewrite_positive(self, claim):
        claim = claim.lower()

        claim = re.sub(r'^key points about', '', claim)
        claim = re.sub(r'^the main points of the discussion, including', '', claim)
        claim = re.sub(r'^the rejection of', '', claim)

        claim = claim.strip()

        return claim[0].upper() + claim[1:] if claim else claim

    def rewrite_negative(self, claim):
        claim = claim.lower()

        claim = re.sub(r'^several important details, such as', '', claim)
        claim = re.sub(r'^several key details, such as', '', claim)
        claim = re.sub(r'^whether', '', claim)

        claim = f"The summary mentions {claim}".strip()

        return claim[0].upper() + claim[1:] if claim else claim