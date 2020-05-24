import java.util.ArrayList;
import java.util.HashSet;

public class AdvancedGinRummyPlayer implements GinRummyPlayer{
	private int playerNum;
	private int startingPlayerNum;
	private int randomSetSize;
	private ArrayList<Card> hand;
	private ArrayList<Card> opponentHand;
	private HashSet<Card> seenCards;
	private Card lastDrawnCard;
	private int deadWood;
	private ArrayList<ArrayList<ArrayList<Card>>> bestMelds;
	private HashSet<Card> wantCards;
	@Override
	public void startGame(int playerNum, int startingPlayerNum, Card[] cards) {
		// TODO Auto-generated method stub
		this.playerNum = playerNum;
		this.startingPlayerNum = startingPlayerNum;
		randomSetSize = 31;
		hand = new ArrayList<Card>();
		seenCards = new HashSet<Card>();
		
		for (Card c: cards) { //hand is  sorted, and all cards in hand added to seen hashset
			seenCards.add(c);
			insertSorted(c, hand);
		}
		
		opponentHand = new ArrayList<Card>();
		updateMeldsDeadWood();
		updateWantCards();
	}

	@Override
	public boolean willDrawFaceUpCard(Card card) {
		// TODO Auto-generated method stub
		//logic to decide whether to take the card from the discarded set
		
		//return true;
		randomSetSize--;
		return false;
	}

	@Override
	public void reportDraw(int playerNum, Card drawnCard) {
		// TODO Auto-generated method stub
		if (playerNum == this.playerNum) { //Reports what we drew
			seenCards.add(drawnCard);
			lastDrawnCard = drawnCard;
			//drawncard is inserted into hand in the proper sorted position
			insertSorted(drawnCard,hand);
			
		}else {
			if (drawnCard == null) { //opponent drew from random set, no knowledge of what the card is
				randomSetSize--;
			}else { //opponent has picked up the card from the discarded set, we have already seen this card before
				insertSorted(drawnCard,opponentHand);
			}
		}
	}

	@Override
	public Card getDiscard() {
		// TODO Auto-generated method stub
		//choose which card to discard, cannot be lastDrawnCard
		updateMeldsDeadWood();
		updateWantCards();
		return null;
	}

	@Override
	public void reportDiscard(int playerNum, Card discardedCard) {
		// TODO Auto-generated method stub
		if (playerNum != this.playerNum) { //reports the card that the opponent discarded
			seenCards.add(discardedCard);
			if (opponentHand.contains(discardedCard))
				opponentHand.remove(discardedCard);
		} //update prediction model???
	}

	@Override
	public ArrayList<ArrayList<Card>> getFinalMelds() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void reportFinalMelds(int playerNum, ArrayList<ArrayList<Card>> melds) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void reportScores(int[] scores) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void reportLayoff(int playerNum, Card layoffCard, ArrayList<Card> opponentMeld) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void reportFinalHand(int playerNum, ArrayList<Card> hand) {
		// TODO Auto-generated method stub
		
	}
	private void insertSorted(Card c, ArrayList<Card> a) {
		if (a.isEmpty())
			a.add(c);
		else {
			int left = 0;
			int right = hand.size() -1;
			int middle = -1;
			while (left <= right) {
				middle = (left + right)/2;
				if (a.get(middle).rank == c.rank) 
					break;
				else if (hand.get(middle).rank < c.rank)
					left = middle + 1;
				else
					right = middle -1;
			}
				if (a.get(middle).rank < c.rank)
					a.add(middle + 1, c);
				else
					a.add(middle,c);
		}
	}
	private void updateMeldsDeadWood() {
		bestMelds = GinRummyUtil.cardsToBestMeldSets(hand);
		deadWood = GinRummyUtil.getDeadwoodPoints(bestMelds.get(0), hand);
	}
	
	private void updateWantCards() {
		wantCards = new HashSet<Card>();
		for (int i = 0; i < hand.size(); ++i) { //this for loop calculates what cards we want to add to sets (if we have 2 or 3 cards of the same rank, look for the last 1 or 2)
			int cardNum = hand.get(i).rank;
			int count = 0;
			HashSet<Integer> suits = new HashSet<Integer>();
			while (i < hand.size() && hand.get(i).rank == cardNum) {
				suits.add(hand.get(i).suit);
				count++;
				i++;
			}
			if (count == 2 || count == 3) {
				if (!suits.contains(0))
					wantCards.add(new Card(cardNum,0));
				if (!suits.contains(1))
					wantCards.add(new Card(cardNum,1));
				if (!suits.contains(2))
					wantCards.add(new Card(cardNum,2));
				if (!suits.contains(3))
					wantCards.add(new Card(cardNum,3));
			}
			i--;
		}
		for (int i = 0; i < hand.size(); ++i) { //this for loop calculates what cards we want to add to runs or form runs
			int suit = hand.get(i).suit;
			int startingRank = hand.get(i).rank;
			int x = i;
			int count = 0;
			ArrayList<Integer> ranks = new ArrayList<Integer>();
			while (x < hand.size() && (hand.get(x).rank == startingRank || hand.get(x).rank == startingRank + 1)) {
				if (ranks.isEmpty()) {
					ranks.add(hand.get(x).rank);
					count++;
				}else {
					if (hand.get(x).suit == suit && hand.get(x).rank == ranks.get(ranks.size() -1)+1) {
						count++;
						ranks.add(hand.get(x).rank);
						startingRank = hand.get(x).rank;
					}
				}
				x++;
			}
			if (count >= 2) {
				if (ranks.get(0) != 0) //the minimal card in a run is an ace
					wantCards.add(new Card(ranks.get(0)-1,suit));
				if (ranks.get(ranks.size() -1) != 12) //the maximal card in a run is a king
					wantCards.add(new Card(ranks.get(ranks.size()-1)+1,suit));
			}else if (count == 1) { //handle the case when there is a card of the same suit in rank startingRank + 2
				startingRank = hand.get(i).rank;
				x = i;
				while (x < hand.size() && hand.get(x).rank <= startingRank + 2) {
					if (hand.get(x).suit == suit && hand.get(x).rank == startingRank + 2) {
						wantCards.add(new Card(startingRank + 1, suit));
						break;
					}
					x++;
				}
			}
		}
	}
	
}