// 
// Copyright (c) 2006-2010, Benjamin Kaufmann
// 
// This file is part of Clasp. See http://www.cs.uni-potsdam.de/clasp/ 
// 
// Clasp is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
// 
// Clasp is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with Clasp; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
//
#ifndef CLASP_SATELITE_H_INCLUDED
#define CLASP_SATELITE_H_INCLUDED

#ifdef _MSC_VER
#pragma once
#endif

#include <clasp/solver.h>
#include <clasp/util/indexed_priority_queue.h>
#include <ctime>

namespace Clasp { namespace SatElite {

//! SatElite preprocessor for clauses
/*!
 * The preprocessor implements subsumption, self-subsumption, variable elimination,
 * and (optionally) blocked clause elimination.
 * \see 
 *   - Niklas E?n, Armin Biere: "Effective Preprocessing in SAT through Variable and Clause Elimination" 
 *   - Matti J?rvisalo, Armin Biere, Marijn Heule: "Blocked Clause Elimination"
 *   - Parts of the SatElite preprocessor are adapted from MiniSAT 2.0 beta
 *     available under the MIT licence from http://minisat.se/MiniSat.html
 *   .
 */
class SatElite : public Clasp::SatPreprocessor {
public:
	SatElite();
	~SatElite();
	Clasp::SatPreprocessor* clone();
	struct Progress : public Event_t<Progress> {
		enum EventOp { event_algorithm = '*', event_bce = 'B', event_var_elim = 'E', event_subsumption = 'S', };
		Progress(SatElite* p, EventOp o, uint32 i, uint32 m) : Event_t<Progress>(Event::subsystem_prepare, Event::verbosity_high), self(p), cur(i), max(m) {
			op = (uint32)o;
		}
		SatElite* self;
		uint32    cur;
		uint32    max;
	};
protected:
	bool  initPreprocess(Options& opts);
	void  reportProgress(Progress::EventOp, uint32 curr, uint32 max);
	bool  doPreprocess();
	void  doExtendModel(ValueVec& m, LitVec& open);
	void  doCleanUp();
private:
	typedef PodVector<uint8>::type    TouchedList;
	typedef bk_lib::left_right_sequence<Literal, Var,0> ClWList;
	typedef ClWList::left_iterator                      ClIter;
	typedef ClWList::right_iterator                     WIter;
	typedef std::pair<ClIter, ClIter>                   ClRange;
	SatElite(const SatElite&);
	const SatElite& operator=(const SatElite&);
	// For each var v
	struct OccurList {
		OccurList() : pos(0), bce(0), dirty(0), neg(0), litMark(0) {}
		ClWList refs;      // left : ids of clauses containing v or ~v  (var() == id, sign() == v or ~v)
		                   // right: ids of clauses watching v or ~v (literal 0 is the watched literal)
		uint32  pos:30;    // number of *relevant* clauses containing v
		uint32  bce:1;     // in BCE queue?
		uint32  dirty:1;   // does clauses contain removed clauses?
		uint32  neg:30;    // number of *relevant* clauses containing v
		uint32  litMark:2; // 00: no literal of v marked, 01: v marked, 10: ~v marked
		uint32  numOcc()          const { return pos + neg; }
		uint32  cost()            const { return pos * neg; }
		ClRange clauseRange()     const { return ClRange(const_cast<ClWList&>(refs).left_begin(), const_cast<ClWList&>(refs).left_end()); }
		void    clear() {
			this->~OccurList();
			new (this) OccurList();
		}
		void    addWatch(uint32 clId)    { refs.push_right(clId); }
		void    removeWatch(uint32 clId) { refs.erase_right(std::find(refs.right_begin(), refs.right_end(), clId)); }
		void    add(uint32 id, bool sign){
			pos += uint32(!sign);
			neg += uint32(sign);
			refs.push_left(Literal(id, sign));
		}
		void    remove(uint32 id, bool sign, bool updateClauseList) {
			pos -= uint32(!sign);
			neg -= uint32(sign);
			if (updateClauseList) { 
				refs.erase_left(std::find(refs.left_begin(), refs.left_end(), Literal(id, sign)));
			}
			else { dirty = 1; }
		}
		// note: only one literal of v shall be marked at a time
		bool    marked(bool sign) const   { return (litMark & (1+int(sign))) != 0; }
		void    mark(bool sign)           { litMark = (1+int(sign)); }
		void    unmark()                  { litMark = 0; }
	};
	struct LessOccCost {
		explicit LessOccCost(OccurList*& occ) : occ_(occ) {}
		bool operator()(Var v1, Var v2) const { return occ_[v1].cost() < occ_[v2].cost(); }
	private:
		const LessOccCost& operator=(LessOccCost&);
		OccurList*& occ_;
	};
	typedef bk_lib::indexed_priority_queue<LessOccCost> ElimHeap;
	Clause*         peekSubQueue() const {
		assert(qFront_ < queue_.size());
		return (Clause*)clause( queue_[qFront_] );
	}
	inline Clause*  popSubQueue();
	inline void     addToSubQueue(uint32 clauseId);
	void            updateHeap(Var v) {
		assert(ctx_);
		if (!ctx_->varInfo(v).frozen() && !ctx_->eliminated(v)) {
			elimHeap_.update(v);
			if (occurs_[v].bce == 0 && occurs_[0].bce != 0) {
				occurs_[0].addWatch(v);
				occurs_[v].bce = 1;
			}
		}
	}
	inline uint32   findUnmarkedLit(const Clause& c, uint32 x) const;
	void    attach(uint32 cId, bool initialClause);
	void    detach(uint32 cId);
	void    bceVeRemove(uint32 cId, bool freeId, Var v, bool blocked);
	bool    propagateFacts();
	bool    backwardSubsume();
	Literal subsumes(const Clause& c, const Clause& other, Literal res) const;
	bool    strengthenClause(uint32 clauseId, Literal p);
	bool    subsumed(LitVec& cl);
	bool    eliminateVars();
	bool    bce();
	bool    bceVe(Var v, uint32 maxCnt);
	ClRange splitOcc(Var v, bool mark);
	bool    trivialResolvent(const Clause& c2, Var v) const;
	void    markAll(const Literal* lits, uint32 size) const;
	void    unmarkAll(const Literal* lits, uint32 size) const;
	bool    addResolvent(uint32 newId, const Clause& c1, const Clause& c2);
	bool    cutoff(Var v) const {
		return opts_->occLimit(occurs_[v].pos, occurs_[v].neg)
		  ||   (occurs_[v].cost() == 0 && opts_->mode == Options::prepro_preserve_models);
	}
	bool    timeout() const { return time(0) > timeout_; }
	const Options*opts_;      // active options
	OccurList*    occurs_;    // occur list for each variable
	ElimHeap      elimHeap_;  // candidates for variable elimination; ordered by increasing occurrence-cost
	VarVec      posT_, negT_; // temporary clause lists used in eliminateVar
	ClauseList    resCands_;  // pairs of clauses to be resolved
	LitVec        resolvent_; // temporary, used in addResolvent
	VarVec        queue_;     // indices of clauses waiting for subsumption-check
	uint32        qFront_;    // front of queue_, i.e. [queue_.begin()+qFront_, queue.end()) is the subsumption queue
	uint32        facts_;     // [facts_, solver.trail.size()): new top-level facts
	std::time_t   timeout_;   // stop once time > timeout_
};
}}
#endif
